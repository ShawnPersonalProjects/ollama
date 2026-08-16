"""The strategy compiler: natural-language mandate -> validated StrategySpec.

This is the heart of the framework: a client describes their execution
strategy in plain English, an LLM compiles it into a machine-readable spec,
and Python validates/clamps every field before the engine will honour it.
"""

import dataclasses
import json
from dataclasses import dataclass

from . import config
from .llm_trader import _strip_code_fences
from .ollama_client import ask_model

STRATEGIES = ("pov", "twap", "opportunistic")
URGENCIES = ("low", "medium", "high")


@dataclass
class StrategySpec:
    strategy: str = "pov"
    side: str = config.DEFAULT_SIDE
    total_qty: int = config.DEFAULT_QTY
    max_pov: float = config.MAX_POV_CAP
    limit_price: float | None = None     # never buy above / sell below
    must_complete: bool = True
    urgency: str = "medium"
    style_notes: str = ""
    mandate_text: str = ""

    def describe(self) -> str:
        limit = f"{self.limit_price:.2f}" if self.limit_price else "none"
        return (f"strategy={self.strategy.upper()} | {self.side.upper()} "
                f"{self.total_qty:,} shares | POV cap {self.max_pov:.0%} | "
                f"limit price: {limit} | must complete: "
                f"{'yes' if self.must_complete else 'no'} | urgency: {self.urgency}\n"
                f"  style: {self.style_notes or '(none)'}")


COMPILE_PROMPT = """You are a trading-strategy compiler. Convert the client's plain-English
execution mandate into a machine-readable JSON spec. Extract only what the
client actually asked for; use the defaults when the mandate is silent.

CLIENT MANDATE:
\"\"\"{mandate}\"\"\"

JSON fields:
- "strategy": one of
    "pov" -- participate in proportion to market volume (default)
    "twap" -- trade at an even pace across the day
    "opportunistic" -- passive baseline, bursts when the price is favourable
- "side": "buy" | "sell" (default "{d_side}")
- "total_qty": integer number of shares (default {d_qty})
- "max_pov": hard participation cap as a fraction 0.01-0.50 (default {d_cap})
- "limit_price": number or null -- hard price limit: never buy above / sell below it (default null)
- "must_complete": true | false -- must the order fully complete today even at bad prices? (default true)
- "urgency": "low" | "medium" | "high" (default "medium")
- "style_notes": one sentence distilling the client's style and priorities,
  written as an instruction to the trading agent

Respond ONLY with the JSON object."""


def spec_from_dict(data: dict, mandate_text: str = "",
                   base: StrategySpec | None = None) -> StrategySpec:
    """Validate and clamp a raw dict into a safe StrategySpec.

    Every field falls back to a sane default -- a hallucinated or malformed
    compile can never produce an unsafe spec. When base is given (an in-flight
    amendment), unspecified/invalid fields keep the current values instead of
    the global defaults.
    """
    spec = dataclasses.replace(base) if base else StrategySpec()
    spec.mandate_text = mandate_text

    strategy = str(data.get("strategy", "")).lower()
    if strategy in STRATEGIES:
        spec.strategy = strategy

    side = str(data.get("side", "")).lower()
    if side in ("buy", "sell"):
        spec.side = side

    qty = data.get("total_qty")
    if isinstance(qty, (int, float)) and qty > 0:
        spec.total_qty = int(qty)

    cap = data.get("max_pov")
    if isinstance(cap, (int, float)) and cap > 0:
        spec.max_pov = min(float(cap), 0.50)  # framework-wide ceiling

    limit = data.get("limit_price")
    if isinstance(limit, (int, float)) and limit > 0:
        spec.limit_price = float(limit)
    elif limit is None and "limit_price" in data:
        spec.limit_price = None  # explicit removal of a limit

    if isinstance(data.get("must_complete"), bool):
        spec.must_complete = data["must_complete"]

    urgency = str(data.get("urgency", "")).lower()
    if urgency in URGENCIES:
        spec.urgency = urgency

    notes = str(data.get("style_notes", "")).strip()
    if notes or base is None:
        spec.style_notes = notes
    return spec


def compile_mandate(mandate: str, model: str = config.MODEL,
                    current: StrategySpec | None = None) -> StrategySpec:
    """Compile a plain-English mandate into a validated StrategySpec via the LLM.

    Pass current for an in-flight amendment: the LLM is told to keep any
    aspect the client did not change, and validation falls back to the
    current spec rather than global defaults.
    """
    prompt = COMPILE_PROMPT.format(
        mandate=mandate.strip(),
        d_side=current.side if current else config.DEFAULT_SIDE,
        d_qty=current.total_qty if current else config.DEFAULT_QTY,
        d_cap=current.max_pov if current else config.MAX_POV_CAP,
    )
    if current:
        limit = f"{current.limit_price:.2f}" if current.limit_price else "null"
        prompt += f"""

NOTE: this is an AMENDMENT to an order already trading. Current spec:
strategy={current.strategy}, side={current.side}, total_qty={current.total_qty},
max_pov={current.max_pov}, limit_price={limit},
must_complete={str(current.must_complete).lower()}, urgency={current.urgency}.
Keep every aspect the client did not explicitly change."""
    raw = ask_model(prompt, model=model)
    try:
        data = json.loads(_strip_code_fences(raw))
        if not isinstance(data, dict):
            data = {}
    except json.JSONDecodeError:
        data = {}
    return spec_from_dict(data, mandate_text=mandate.strip(), base=current)
