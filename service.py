"""Internal research forecast API. Configure MODEL_DIR and FORECAST_API_TOKEN."""

import hashlib
import hmac
import json
import logging
import os
import statistics
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import lightgbm as lgb
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from features import FEATURE_NAMES, MIN_HISTORY, validate_history
from research import METHODS, compare_methods, rolling_backtest, select_method

logger = logging.getLogger("forecast")


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date
    units_sold: StrictInt = Field(ge=0, le=1_000_000)
    stockout: StrictBool | None


class VariantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant_id: str = Field(min_length=1, max_length=100)
    history: list[HistoryPoint] = Field(max_length=730)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=100)
    as_of: date
    min_series_days: StrictInt = Field(default=28, ge=28, le=730)
    variants: list[VariantRequest] = Field(min_length=1, max_length=200)


def load_model(directory: Path) -> tuple[dict, list]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_names"] != list(FEATURE_NAMES) or metadata["dataset_kind"] != "synthetic":
        raise ValueError("model feature schema or dataset kind does not match service")
    models = []
    for index in range(3):
        name = f"segment-{index}.txt"
        path = directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["checksums"][name]:
            raise ValueError(f"model checksum mismatch: {name}")
        models.append(lgb.Booster(model_file=str(path)))
    return metadata, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.error = "MODEL_DIR is not configured"
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        try:
            app.state.model = load_model(Path(model_dir))
            app.state.error = None
        except (OSError, ValueError, KeyError, json.JSONDecodeError, lgb.basic.LightGBMError) as error:
            app.state.error = f"Model could not be loaded: {error}"
            logger.error("%s", app.state.error)
    yield


app = FastAPI(title="POS forecasting research", version="1", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    model = request.app.state.model
    return {
        "ready": model is not None and bool(os.environ.get("FORECAST_API_TOKEN")),
        "model_version": model[0]["model_version"] if model else None,
    }


@app.post("/v1/forecasts:predict")
def forecasts(body: ForecastRequest, request: Request, x_forecast_token: str | None = Header(default=None)):
    token = os.environ.get("FORECAST_API_TOKEN")
    if not token:
        raise HTTPException(503, "Forecast service authentication is not configured")
    if not x_forecast_token or not hmac.compare_digest(token.encode(), x_forecast_token.encode()):
        raise HTTPException(401, "Invalid forecast service token")
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Forecast model is unavailable")
    if len({item.variant_id for item in body.variants}) != len(body.variants):
        raise HTTPException(422, "Duplicate variant_id")
    metadata, models = model
    results = []
    for item in body.variants:
        rows = [row.model_dump(mode="json") for row in item.history]
        if not rows:
            results.append({"variant_id": item.variant_id, "status": "INSUFFICIENT_DATA", "series_days": 0, "reason": "SHORT_HISTORY"})
            continue
        try:
            validate_history(rows, body.as_of)
        except (ValueError, KeyError) as error:
            results.append({"variant_id": item.variant_id, "status": "INVALID_HISTORY", "message": str(error)})
            continue
        if len(rows) < max(MIN_HISTORY, body.min_series_days):
            results.append({"variant_id": item.variant_id, "status": "INSUFFICIENT_DATA", "series_days": len(rows), "reason": "SHORT_HISTORY"})
            continue
        clean = [row["units_sold"] for row in rows[-28:] if row["stockout"] is False]
        if len(clean) < 7 or any(row["stockout"] is None for row in rows[-28:]):
            results.append({"variant_id": item.variant_id, "status": "INSUFFICIENT_DATA", "series_days": len(rows), "reason": "CENSORED_HISTORY"})
            continue
        last_unknown = next((index for index in range(len(rows) - 1, -1, -1) if rows[index]["stockout"] is None), -1)
        known_history = rows[last_unknown + 1:]
        impute = metadata["impute_censored_history"]
        scores = rolling_backtest(known_history, models, impute)
        selected = select_method(scores)
        predictions = compare_methods(models, known_history, impute)
        totals = predictions[selected]
        score = scores[selected]
        recent = rows[-28:]
        active_share = sum(row["units_sold"] > 0 for row in recent) / len(recent)
        stockout_share = sum(row["stockout"] is True for row in recent) / len(recent)
        wape = score["wape"]
        confidence = None
        if score["count"] >= 2 and wape is not None:
            confidence = "HIGH" if wape <= 0.35 and active_share >= 0.5 and stockout_share < 0.1 else (
                "MEDIUM" if wape <= 0.7 and active_share >= 0.2 else "LOW"
            )
        results.append({
            "variant_id": item.variant_id,
            "status": "OK",
            "series_days": len(rows),
            "model": selected.upper(),
            "daily_sigma": statistics.stdev(clean),
            "segments": {
                "days_1_7": totals[0],
                "days_8_14": totals[1] - totals[0],
                "days_15_28": totals[2] - totals[1],
            },
            "totals": {"d7": totals[0], "d14": totals[1], "d28": totals[2]},
            "confidence": confidence,
            "wape": wape,
            "bias": score["bias"],
            "active_share": active_share,
            "stockout_share": stockout_share,
            "comparison": [
                {"model": method.upper(), "totals": dict(zip(("d7", "d14", "d28"), predictions[method])), **scores[method]}
                for method in METHODS
            ],
            "warnings": ["MODEL_TRAINED_ON_SYNTHETIC_DATA"],
        })
    return {
        "request_id": body.request_id,
        "as_of": body.as_of.isoformat(),
        "model_version": metadata["model_version"],
        "dataset_kind": metadata["dataset_kind"],
        "stock_history_policy": "known-v1",
        "results": results,
    }
