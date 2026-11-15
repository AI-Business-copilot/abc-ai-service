import unittest
from datetime import date
from unittest.mock import patch

from research import metrics, rolling_backtest, select_method, simulate_policy, train_rows
from synthetic import generate


class ResearchTests(unittest.TestCase):
    def test_metrics_does_not_invent_wape_when_no_sales(self):
        result = metrics([(0, 3), (0, 1)])
        self.assertEqual(result["mae"], 2)
        self.assertIsNone(result["wape"])
        self.assertIsNone(result["bias"])

    def test_training_does_not_use_rows_after_cutoff_or_hidden_demand(self):
        variants = generate(1, 24, 180, date(2025, 1, 1))["variants"]
        before_x, before_y = train_rows(variants, 120, True)
        for variant in variants:
            for row in variant["days"]:
                row["latent_demand"] += 5000
                if row["date"] >= "2025-05-01":
                    row["units_sold"] += 5000
        after_x, after_y = train_rows(variants, 120, True)
        self.assertTrue((before_x == after_x).all())
        for before, after in zip(before_y, after_y):
            self.assertTrue((before == after).all())

    def test_policy_compares_same_latent_demand_and_respects_lead_time(self):
        future = [{"latent_demand": 1} for _ in range(7)]
        result = simulate_policy(0, future, 1)
        self.assertEqual(result["lost_units"], 5)
        self.assertEqual(result["stockout_days"], 5)
        self.assertEqual(result["ending_stock"], 14)

    def test_rolling_backtest_uses_same_uncensored_windows_and_past_only(self):
        rows = [{"units_sold": 2, "stockout": False} for _ in range(56)]
        rows[42]["stockout"] = True
        seen = []

        def fake_predict(_models, history, _impute):
            seen.append(len(history))
            return [14, 28, 56]

        with patch("research.predict", side_effect=fake_predict):
            scores = rolling_backtest(rows, [], True)
        self.assertEqual(seen, [49, 35, 28])
        self.assertTrue(all(score["count"] == 3 for score in scores.values()))
        self.assertEqual(scores["mean_28"]["wape"], 0)

    def test_selector_requires_two_windows_and_material_improvement(self):
        scores = {
            "mean_28": {"count": 1, "wape": 1.0},
            "ses": {"count": 1, "wape": 0.2},
            "croston_sba": {"count": 1, "wape": 0.5},
            "lightgbm": {"count": 1, "wape": 0.8},
        }
        self.assertEqual(select_method(scores), "mean_28")
        for score in scores.values():
            score["count"] = 2
        self.assertEqual(select_method(scores), "ses")
        scores["ses"]["wape"] = 0.95
        scores["croston_sba"]["wape"] = 0.95
        scores["lightgbm"]["wape"] = 0.95
        self.assertEqual(select_method(scores), "mean_28")

    def test_backtest_skips_origins_with_unknown_recent_stock_history(self):
        rows = [{"units_sold": 2, "stockout": False} for _ in range(56)]
        rows[27]["stockout"] = None
        with patch("research.predict", return_value=[14, 28, 56]):
            scores = rolling_backtest(rows, [], True)
        self.assertTrue(all(score["count"] == 0 for score in scores.values()))


if __name__ == "__main__":
    unittest.main()
