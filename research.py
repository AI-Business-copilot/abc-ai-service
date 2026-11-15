"""Train fixed candidate models and evaluate on chronologically held-out synthetic data."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES, MIN_HISTORY, feature_vector, observed_values

SEGMENTS = ((0, 7), (7, 14), (14, 28))
HORIZONS = (7, 14, 28)
METHODS = ("mean_28", "ses", "croston_sba", "lightgbm")


def baseline(history: list[dict], method: str, impute: bool) -> float:
    values = observed_values(history, impute)
    if method == "mean_28":
        return sum(values[-28:]) / 28
    if method == "ses":
        level = sum(values[: min(7, len(values))]) / min(7, len(values))
        for value in values:
            level = 0.3 * value + 0.7 * level
        return level
    if method == "croston_sba":
        size = interval = None
        gap = 1
        for value in values:
            if value > 0:
                size = value if size is None else size + 0.1 * (value - size)
                interval = gap if interval is None else interval + 0.1 * (gap - interval)
                gap = 1
            else:
                gap += 1
        return 0.0 if size is None else 0.95 * size / interval
    raise ValueError(f"unknown baseline {method}")


def train_rows(variants: list[dict], train_end: int, impute: bool) -> tuple:
    features, targets = [], [[], [], []]
    for variant in variants:
        rows = variant["days"]
        for end in range(MIN_HISTORY, train_end - 28 + 1, 7):
            future = rows[end : end + 28]
            # Censored labels are not treated as actual demand during training.
            if len(future) != 28 or any(row["stockout"] is not False for row in future):
                continue
            features.append(feature_vector(rows[:end], impute))
            for index, (start, stop) in enumerate(SEGMENTS):
                targets[index].append(sum(row["units_sold"] for row in future[start:stop]))
    if len(features) < 20:
        raise ValueError("not enough uncensored training windows; increase dataset size or stock availability")
    return np.asarray(features, dtype=float), [np.asarray(target, dtype=float) for target in targets]


def train_models(features: np.ndarray, targets: list[np.ndarray], seed: int) -> list:
    models = []
    for target in targets:
        model = lgb.train(
            {
                "objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
                "min_data_in_leaf": 15, "seed": seed, "num_threads": 1,
                "verbosity": -1, "deterministic": True, "force_col_wise": True,
            },
            lgb.Dataset(features, label=target, feature_name=list(FEATURE_NAMES)),
            num_boost_round=100,
        )
        models.append(model)
    return models


def predict(models: list, history: list[dict], impute: bool) -> list[float]:
    features = np.asarray([feature_vector(history, impute)], dtype=float)
    segments = [float(model.predict(features, num_threads=1)[0]) for model in models]
    if not all(math.isfinite(value) for value in segments):
        raise ValueError("model returned a non-finite forecast")
    segments = [max(0.0, value) for value in segments]
    return [segments[0], segments[0] + segments[1], sum(segments)]


def predict_method(method: str, models: list, history: list[dict], impute: bool) -> list[float]:
    if method == "lightgbm":
        return predict(models, history, impute)
    rate = baseline(history, method, impute)
    return [rate * horizon for horizon in HORIZONS]


def compare_methods(models: list, history: list[dict], impute: bool) -> dict:
    return {method: predict_method(method, models, history, impute) for method in METHODS}


def rolling_backtest(history: list[dict], models: list, impute: bool) -> dict:
    """Compare all methods on the same fully observed seven-day windows."""
    pairs = {method: [] for method in METHODS}
    for offset in range(1, 5):
        cut = len(history) - 7 * offset
        if cut < MIN_HISTORY:
            break
        future = history[cut:cut + 7]
        if (any(row.get("stockout") is None for row in history[:cut])
                or any(row.get("stockout") is not False for row in future)):
            continue
        actual = sum(row["units_sold"] for row in future)
        for method, forecast in compare_methods(models, history[:cut], impute).items():
            pairs[method].append((actual, forecast[0]))
    return {method: metrics(values) for method, values in pairs.items()}


def select_method(scores: dict) -> str:
    """Require two windows and a 10% improvement over the mean before switching."""
    baseline_score = scores["mean_28"]
    if baseline_score["count"] < 2 or baseline_score["wape"] is None:
        return "mean_28"
    best = "mean_28"
    best_wape = baseline_score["wape"]
    for method in METHODS[1:]:
        score = scores[method]["wape"]
        if score is not None and score < baseline_score["wape"] * 0.9 and score < best_wape:
            best, best_wape = method, score
    return best


def metrics(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"count": 0, "mae": None, "wape": None, "bias": None}
    actual = sum(a for a, _ in pairs)
    absolute = sum(abs(p - a) for a, p in pairs)
    signed = sum(p - a for a, p in pairs)
    return {
        "count": len(pairs),
        "mae": absolute / len(pairs),
        "wape": absolute / actual if actual > 0 else None,
        "bias": signed / actual if actual > 0 else None,
    }


def simulate_policy(initial_stock: int, future: list[dict], daily_rate: float) -> dict:
    """Compare forecasts with the same deliberately simplified replenishment policy."""
    stock = initial_stock
    on_order = []
    lost = stockout_days = holding_unit_days = 0
    for day, row in enumerate(future):
        arrivals = sum(qty for arrival, qty in on_order if arrival == day)
        on_order = [(arrival, qty) for arrival, qty in on_order if arrival > day]
        stock += arrivals
        if day % 7 == 0:
            position = stock + sum(qty for _, qty in on_order)
            reorder = daily_rate * 5 + 4
            target = daily_rate * 12 + 4
            if position <= reorder and target > position:
                on_order.append((day + 5, math.ceil(target - position)))
        demand = row["latent_demand"]
        sold = min(stock, demand)
        stock -= sold
        lost += demand - sold
        stockout_days += int(demand > sold)
        holding_unit_days += stock
    return {
        "lost_units": lost, "stockout_days": stockout_days,
        "ending_stock": stock, "holding_unit_days": holding_unit_days,
    }


def evaluate(variants: list[dict], models: list, start: int, end: int, impute: bool) -> dict:
    results = {}
    policy_results = {}
    for variant in variants:
        rows = variant["days"]
        for origin in range(start, end - 28 + 1, 28):
            history = rows[:origin]
            future = rows[origin : origin + 28]
            predictions = compare_methods(models, history, impute)
            for method, values in predictions.items():
                outcome = simulate_policy(history[-1]["stock_end_of_day"], future, values[2] / 28)
                for group in ("all", variant["scenario"]):
                    key = (method, group)
                    entry = policy_results.setdefault(key, {"count": 0, "lost_units": 0, "stockout_days": 0, "ending_stock": 0, "holding_unit_days": 0})
                    entry["count"] += 1
                    for field, value in outcome.items():
                        entry[field] += value
            for index, horizon in enumerate(HORIZONS):
                window = future[:horizon]
                clean = all(row["stockout"] is False for row in window)
                latent = sum(row["latent_demand"] for row in window)
                observed = sum(row["units_sold"] for row in window)
                for method, values in predictions.items():
                    # Both overall and scenario scores are reported; latent demand is evaluation-only.
                    for group in ("all", variant["scenario"]):
                        key = (method, str(horizon), group)
                        entry = results.setdefault(key, {"latent": [], "observed_uncensored": []})
                        entry["latent"].append((latent, values[index]))
                        if clean:
                            entry["observed_uncensored"].append((observed, values[index]))
    return {
        "forecast": [
            {
                "model": model, "horizon": int(horizon), "group": group,
                "latent": metrics(values["latent"]),
                "observed_uncensored": metrics(values["observed_uncensored"]),
            }
            for (model, horizon, group), values in sorted(results.items())
        ],
        "policy": [
            {"model": model, "group": group, **values}
            for (model, group), values in sorted(policy_results.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory for model and report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-impute", action="store_true")
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    variants = dataset["variants"]
    if not variants:
        raise ValueError("dataset has no variants")
    days = len(variants[0]["days"])
    if days < 180 or any(len(v["days"]) != days for v in variants):
        raise ValueError("research dataset must have equal histories of at least 180 days")
    train_end = math.floor(days * 0.60)
    validation_end = math.floor(days * 0.80)
    heldout = [v for i, v in enumerate(variants) if i % 5 == 0]
    seen = [v for i, v in enumerate(variants) if i % 5 != 0]
    x, y = train_rows(seen, train_end, not args.no_impute)
    models = train_models(x, y, args.seed)
    report = {
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_seed": dataset["seed"],
        "model_seed": args.seed,
        "lightgbm_version": lgb.__version__,
        "numpy_version": np.__version__,
        "train_end_index_exclusive": train_end,
        "validation_end_index_exclusive": validation_end,
        "test_end_index_exclusive": days,
        "train_windows": len(x),
        "impute_censored_history": not args.no_impute,
        "model_selection": "fixed candidate; no tuning on validation or test",
        "policy_assumptions": "independent 28-day simulation; lead 5 days, review 7 days, safety 4 units, no initial orders or returns, constant d28/28 demand estimate",
        "validation_seen_skus": evaluate(seen, models, train_end, validation_end, not args.no_impute),
        "test_seen_skus": evaluate(seen, models, validation_end, days, not args.no_impute),
        "test_heldout_skus": evaluate(heldout, models, validation_end, days, not args.no_impute),
    }
    args.out.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for index, model in enumerate(models):
        name = f"segment-{index}.txt"
        model.save_model(str(args.out / name))
        checksums[name] = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
    metadata = {
        "model_version": f"synthetic-lgbm-{args.seed}-{report['dataset_sha256'][:12]}-{'raw' if args.no_impute else 'imputed'}",
        "dataset_kind": "synthetic",
        "feature_names": list(FEATURE_NAMES),
        "impute_censored_history": not args.no_impute,
        "checksums": checksums,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Trained on {len(x)} windows; model and report: {args.out}")


if __name__ == "__main__":
    main()
