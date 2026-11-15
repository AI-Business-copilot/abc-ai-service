"""Feature extraction from observed POS history only."""

from datetime import date, timedelta
from math import sqrt

MIN_HISTORY = 28
FEATURE_NAMES = (
    "mean_7", "mean_14", "mean_28", "std_28",
    "active_28", "stockout_28", "unknown_stockout_28", "trend_7",
)


def validate_history(history: list[dict], as_of: date) -> None:
    if not history:
        raise ValueError("history is empty")
    last = None
    for row in history:
        day = date.fromisoformat(row["date"])
        if last is not None and day != last + timedelta(days=1):
            raise ValueError("history dates must be consecutive and strictly increasing")
        value = row["units_sold"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("units_sold must be a non-negative integer")
        if row.get("stockout") not in (True, False, None):
            raise ValueError("stockout must be true, false or null")
        last = day
    if last != as_of:
        raise ValueError("history must end on as_of")


def observed_values(history: list[dict], impute: bool = True) -> list[float]:
    values = [float(row["units_sold"]) for row in history]
    clean = [value for value, row in zip(values, history) if row.get("stockout") is False]
    if impute and len(clean) >= 7:
        replacement = sum(clean) / len(clean)
        values = [replacement if row.get("stockout") is True else value for value, row in zip(values, history)]
    return values


def feature_vector(history: list[dict], impute: bool = True) -> list[float]:
    if len(history) < MIN_HISTORY:
        raise ValueError("at least 28 days of history are required")
    window = history[-28:]
    values = observed_values(window, impute)
    mean = lambda n: sum(values[-n:]) / n
    avg = mean(28)
    std = sqrt(sum((value - avg) ** 2 for value in values) / 27)
    return [
        mean(7), mean(14), avg, std,
        sum(row["units_sold"] > 0 for row in window) / 28,
        sum(row.get("stockout") is True for row in window) / 28,
        sum(row.get("stockout") is None for row in window) / 28,
        mean(7) - sum(values[-14:-7]) / 7,
    ]
