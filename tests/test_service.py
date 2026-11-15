import os
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from service import ForecastRequest, forecasts
from synthetic import generate


class ServiceTests(unittest.TestCase):
    def setUp(self):
        days = generate(4, 1, 28, date(2025, 1, 1))["variants"][0]["days"]
        self.history = [
            {"date": row["date"], "units_sold": row["units_sold"], "stockout": False}
            for row in days
        ]
        self.metadata = {
            "model_version": "test", "dataset_kind": "synthetic",
            "impute_censored_history": True,
        }
        self.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model=(self.metadata, []))))

    def test_authentication_and_consistent_totals(self):
        body = ForecastRequest(
            request_id="test", as_of="2025-01-28",
            variants=[{"variant_id": "sku", "history": self.history}],
        )
        with patch.dict(os.environ, {"FORECAST_API_TOKEN": "local-test"}):
            with self.assertRaises(HTTPException) as error:
                forecasts(body, self.request, "wrong")
            self.assertEqual(error.exception.status_code, 401)
            with patch("research.predict", return_value=[7.0, 15.0, 31.0]):
                result = forecasts(body, self.request, "local-test")["results"][0]
        self.assertEqual(result["model"], "MEAN_28")
        self.assertEqual(sum(result["segments"].values()), result["totals"]["d28"])
        self.assertIsNone(result["confidence"])
        self.assertEqual(len(result["comparison"]), 4)
        self.assertEqual(next(row for row in result["comparison"] if row["model"] == "LIGHTGBM")["totals"]["d28"], 31)

    def test_short_and_invalid_history_are_not_predicted(self):
        short = self.history[-10:]
        invalid = self.history[:]
        del invalid[5]
        body = ForecastRequest(
            request_id="test", as_of="2025-01-28",
            variants=[
                {"variant_id": "short", "history": short},
                {"variant_id": "invalid", "history": invalid},
            ],
        )
        with patch.dict(os.environ, {"FORECAST_API_TOKEN": "local-test"}), patch("service.compare_methods") as predict:
            result = forecasts(body, self.request, "local-test")["results"]
        self.assertEqual([row["status"] for row in result], ["INSUFFICIENT_DATA", "INVALID_HISTORY"])
        predict.assert_not_called()

    def test_empty_or_heavily_censored_history_is_insufficient(self):
        body = ForecastRequest(
            request_id="test", as_of="2025-01-28",
            variants=[
                {"variant_id": "new", "history": []},
                {"variant_id": "out", "history": [{**row, "stockout": True} for row in self.history]},
            ],
        )
        with patch.dict(os.environ, {"FORECAST_API_TOKEN": "local-test"}):
            results = forecasts(body, self.request, "local-test")["results"]
        self.assertEqual([row["reason"] for row in results], ["SHORT_HISTORY", "CENSORED_HISTORY"])

    def test_rejects_latent_demand_in_api(self):
        with self.assertRaises(ValidationError):
            ForecastRequest(
                request_id="test", as_of="2025-01-28",
                variants=[{"variant_id": "sku", "history": [{**self.history[-1], "latent_demand": 5}]}],
            )


if __name__ == "__main__":
    unittest.main()
