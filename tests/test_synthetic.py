import unittest
from datetime import date

from features import feature_vector, validate_history
from synthetic import generate


class SyntheticTests(unittest.TestCase):
    def test_reproducible_and_stock_balances(self):
        first = generate(7, 12, 70, date(2025, 1, 1))
        self.assertEqual(first, generate(7, 12, 70, date(2025, 1, 1)))
        self.assertNotEqual(first, generate(8, 12, 70, date(2025, 1, 1)))
        for variant in first["variants"]:
            stock = variant["opening_stock"]
            for row in variant["days"]:
                stock += row["received_qty"] + row["units_returned"] - row["scrapped_qty"] - row["units_sold"]
                self.assertEqual(stock, row["stock_end_of_day"])
                self.assertGreaterEqual(stock, 0)
                self.assertEqual(row["units_sold"] + row["lost_demand"], row["latent_demand"])

    def test_features_do_not_read_latent_demand(self):
        days = generate(7, 1, 28, date(2025, 1, 1))["variants"][0]["days"]
        expected = feature_vector(days)
        for row in days:
            row["latent_demand"] = 1000000
            row["lost_demand"] = 1000000
        self.assertEqual(feature_vector(days), expected)

    def test_rejects_missing_dates_instead_of_filling_zero(self):
        days = generate(7, 1, 30, date(2025, 1, 1))["variants"][0]["days"]
        del days[10]
        with self.assertRaisesRegex(ValueError, "consecutive"):
            validate_history(days, date.fromisoformat(days[-1]["date"]))


if __name__ == "__main__":
    unittest.main()
