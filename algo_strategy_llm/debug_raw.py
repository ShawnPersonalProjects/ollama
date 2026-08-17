"""One-off debug: print the raw LLM output for a sample decision prompt.

Run: python -m pov_strategy.debug_raw
"""

from .llm_trader import build_decision_prompt
from .ollama_client import ask_model

state = {
    "side": "buy",
    "total_qty": 1_000_000,
    "filled_qty": 220_000,
    "remaining_qty": 780_000,
    "max_pov": 0.25,
    "bar_index": 120,
    "n_bars": 390,
    "arrival_price": 100.0,
    "last_price": 99.60,
    "market_vwap": 100.10,
    "recent_bars": [(99.7, 12000), (99.65, 11000), (99.6, 13000)],
    "avg_recent_volume": 12000,
    "avg_day_volume": 12800,
    "realized_pov": 0.14,
    "required_pov": 0.226,
}

raw = ask_model(build_decision_prompt(state))
print("RAW OUTPUT:")
print(repr(raw))
