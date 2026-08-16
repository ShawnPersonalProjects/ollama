"""POV execution engine: turns LLM decisions into simulated child-order fills."""

from dataclasses import dataclass
from datetime import datetime

from .llm_trader import Decision
from .market_data import Bar

# Simplified microstructure model per aggressiveness level:
#   fill_ratio  -- fraction of the intended child quantity actually captured
#   penalty_bps -- adverse price paid relative to the bar close
FILL_MODEL = {
    "passive": {"fill_ratio": 0.60, "penalty_bps": 0.5},
    "neutral": {"fill_ratio": 0.90, "penalty_bps": 2.0},
    "aggressive": {"fill_ratio": 1.00, "penalty_bps": 5.0},
}


@dataclass
class Fill:
    bar_index: int
    ts: datetime
    qty: int
    price: float
    bar_volume: int
    target_pov: float
    aggressiveness: str


class POVExecutor:
    """Executes a parent order bar-by-bar, honouring hard guardrails.

    Supports three pacing strategies: "pov" (participation-based), "twap"
    (even schedule x LLM pace multiplier), "opportunistic" (participation-based;
    behavioural difference lives in the LLM prompt).
    """

    def __init__(self, total_qty: int, side: str, max_pov: float,
                 limit_price: float | None = None, must_complete: bool = True,
                 strategy: str = "pov", n_bars: int | None = None):
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        self.total_qty = total_qty
        self.side = side
        self.max_pov = max_pov
        self.limit_price = limit_price
        self.must_complete = must_complete
        self.strategy = strategy
        self.n_bars = n_bars

        self.filled_qty = 0
        self.fill_notional = 0.0
        self.fills: list[Fill] = []
        self.failsafe_engaged = False
        self.limit_blocked_bars = 0

        # Running market stats (over bars seen while the order is active).
        self.market_volume = 0
        self.market_notional = 0.0
        self.arrival_price: float | None = None

    @property
    def remaining_qty(self) -> int:
        return self.total_qty - self.filled_qty

    @property
    def done(self) -> bool:
        return self.remaining_qty <= 0

    @property
    def avg_fill_price(self) -> float:
        return self.fill_notional / self.filled_qty if self.filled_qty else 0.0

    @property
    def market_vwap(self) -> float:
        return self.market_notional / self.market_volume if self.market_volume else 0.0

    @property
    def realized_pov(self) -> float:
        return self.filled_qty / self.market_volume if self.market_volume else 0.0

    def execute_bar(self, bar_index: int, bar: Bar, decision: Decision,
                    est_future_volume: float | None = None) -> Fill | None:
        """Execute one bar under the given decision. Returns the fill, if any.

        est_future_volume: expected market volume from this bar to the close.
        When completing on time requires participation at or above the cap,
        a must-complete failsafe overrides the LLM with cap + aggressive.
        """
        if self.arrival_price is None:
            self.arrival_price = bar.open

        self.market_volume += bar.volume
        self.market_notional += bar.volume * bar.close

        if self.done:
            return None

        # Hard limit price: never buy above / sell below the client's limit.
        if self.limit_price is not None:
            if (self.side == "buy" and bar.close > self.limit_price) or \
               (self.side == "sell" and bar.close < self.limit_price):
                self.limit_blocked_bars += 1
                return None

        # Guardrails: clamp POV to the cap and never exceed the remainder.
        pov = max(0.0, min(decision.target_pov, self.max_pov))
        aggressiveness = decision.aggressiveness

        # Must-complete failsafe: override the LLM when it can no longer pace.
        if self.must_complete and est_future_volume and est_future_volume > 0:
            required_pov = self.remaining_qty / est_future_volume
            if required_pov >= self.max_pov:
                pov, aggressiveness = self.max_pov, "aggressive"
                self.failsafe_engaged = True

        model = FILL_MODEL.get(aggressiveness, FILL_MODEL["neutral"])

        if self.strategy == "twap" and self.n_bars and not self.failsafe_engaged:
            # Even schedule x LLM pace multiplier, still capped by max POV.
            bars_left = max(1, self.n_bars - bar_index)
            intended = (self.remaining_qty / bars_left) * decision.pace * model["fill_ratio"]
            intended = min(intended, self.max_pov * bar.volume)
        else:
            intended = pov * bar.volume * model["fill_ratio"]

        qty = min(round(intended), self.remaining_qty)
        if qty <= 0:
            return None

        # Adverse price: buys pay up, sells give up.
        sign = 1 if self.side == "buy" else -1
        price = bar.close * (1 + sign * model["penalty_bps"] / 10_000)

        fill = Fill(
            bar_index, bar.ts, qty, price, bar.volume,
            pov, aggressiveness,
        )
        self.fills.append(fill)
        self.filled_qty += qty
        self.fill_notional += qty * price
        return fill
