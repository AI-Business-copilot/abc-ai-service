"""Reproducible, isolated research data. Latent demand is never a model feature."""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

SCENARIOS = ("steady", "intermittent", "rising", "falling", "volatile", "stockout")


def poisson(rng: random.Random, rate: float) -> int:
    # Small rates are sufficient for the simulated shop SKUs.
    import math

    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def generate(seed: int, sku_count: int, days: int, start: date) -> dict:
    if sku_count < 1 or days < 1:
        raise ValueError("sku_count and days must be positive")
    rng = random.Random(seed)
    variants = []
    for index in range(sku_count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        base = rng.uniform(0.7, 4.0)
        stock = rng.randint(8, 40)
        opening = stock
        previous_sales = 0
        rows = []
        for day in range(days):
            received = rng.randint(12, 30) if day % 14 == 0 and scenario != "stockout" else 0
            returned = 1 if previous_sales > 0 and rng.random() < 0.02 else 0
            scrapped = 1 if stock + received + returned > 0 and rng.random() < 0.005 else 0
            stock += received + returned - scrapped
            rate = base
            if scenario == "intermittent":
                rate = base if rng.random() < 0.2 else 0
            elif scenario == "rising":
                rate = base * (0.5 + day / max(1, days))
            elif scenario == "falling":
                rate = base * (1.5 - day / max(1, days))
            elif scenario == "volatile":
                rate = base * rng.uniform(0.2, 1.8)
            elif scenario == "stockout" and day % 35 == 0:
                received = rng.randint(10, 25)
                stock += received
            latent = poisson(rng, rate) if rate > 0 else 0
            sold = min(stock, latent)
            stock -= sold
            rows.append(
                {
                    "date": (start + timedelta(days=day)).isoformat(),
                    "units_sold": sold,
                    "received_qty": received,
                    "units_returned": returned,
                    "scrapped_qty": scrapped,
                    "stock_end_of_day": stock,
                    "stockout": stock == 0,
                    "latent_demand": latent,
                    "lost_demand": latent - sold,
                }
            )
            previous_sales = sold
        variants.append(
            {
                "variant_id": f"synthetic-{index:05d}",
                "category": "clothing" if index % 2 == 0 else "cosmetics",
                "scenario": scenario,
                "opening_stock": opening,
                "days": rows,
            }
        )
    return {
        "generator_version": 1,
        "seed": seed,
        "start": start.isoformat(),
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an isolated synthetic POS research dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skus", type=int, default=300)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 1))
    args = parser.parse_args()
    dataset = generate(args.seed, args.skus, args.days, args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, separators=(",", ":"))
    print(f"Generated {args.skus} SKUs x {args.days} days: {args.out}")


if __name__ == "__main__":
    main()
