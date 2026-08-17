"""Market data: synthetic intraday bar generator + optional yfinance loader."""

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import config


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def _u_shape_weight(t: float) -> float:
    """Intraday volume weight: heavy at open/close, light midday.

    t is the fraction of the day elapsed, in [0, 1].
    """
    return 1.0 + 1.5 * (2.0 * t - 1.0) ** 2


def generate_synthetic_day(
    n_bars: int = config.SYNTHETIC_BARS,
    start_price: float = config.SYNTHETIC_START_PRICE,
    daily_vol: float = config.SYNTHETIC_DAILY_VOL,
    day_volume: int = config.SYNTHETIC_DAY_VOLUME,
    seed: int | None = None,
) -> list[Bar]:
    """Generate one trading day of 1-minute bars.

    Price follows a geometric Brownian motion; volume follows a noisy
    U-shaped intraday profile.
    """
    rng = random.Random(seed)
    bar_vol = daily_vol / math.sqrt(n_bars)  # per-bar volatility

    # Volume profile: U-shape * lognormal noise, scaled to day_volume.
    raw_weights = [
        _u_shape_weight(i / (n_bars - 1)) * rng.lognormvariate(0, 0.35)
        for i in range(n_bars)
    ]
    total_weight = sum(raw_weights)

    bars: list[Bar] = []
    ts = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    price = start_price
    for i in range(n_bars):
        open_ = price
        ret = rng.gauss(0, bar_vol)
        close = open_ * math.exp(ret)
        wiggle = abs(rng.gauss(0, bar_vol / 2))
        high = max(open_, close) * (1 + wiggle)
        low = min(open_, close) * (1 - wiggle)
        volume = max(1, round(day_volume * raw_weights[i] / total_weight))
        bars.append(Bar(ts, open_, high, low, close, volume))
        price = close
        ts += timedelta(minutes=1)
    return bars


def load_yfinance(symbol: str, interval: str = "1m", period: str = "1d") -> list[Bar]:
    """Load real intraday bars via yfinance (optional dependency)."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Run: pip install yfinance\n"
            "Or use --source synthetic instead."
        )

    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        raise SystemExit(f"No data returned for '{symbol}' ({interval}, {period}).")
    if hasattr(df.columns, "levels"):  # flatten MultiIndex columns if present
        df.columns = df.columns.get_level_values(0)

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        volume = int(row["Volume"])
        if volume <= 0:
            continue
        bars.append(
            Bar(
                ts.to_pydatetime(),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                volume,
            )
        )
    if not bars:
        raise SystemExit(f"No bars with volume for '{symbol}'.")
    return bars
