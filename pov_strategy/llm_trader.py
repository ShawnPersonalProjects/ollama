"""LLM decision-making: prompt construction and robust decision parsing."""

import json
import re
from dataclasses import dataclass

from . import config

AGGRESSIVENESS_LEVELS = ("passive", "neutral", "aggressive")


@dataclass
class Decision:
    target_pov: float
    aggressiveness: str
    reasoning: str
    pace: float = 1.0  # twap only: multiplier on the even schedule


def neutral_default(reason: str = "fallback: unusable LLM output") -> Decision:
    return Decision(config.DEFAULT_POV, "neutral", reason)


_STRATEGY_GUIDANCE = {
    "pov": """- You are running a POV strategy: participate in proportion to market volume.
- Higher target_pov and "aggressive" complete faster but pay more spread/impact.
- Go "passive" with lower pov only when the price is unfavourable, plenty of time
  remains, AND you are comfortably ahead of the minimum required POV.""",
    "twap": """- You are running a TWAP strategy: trade at an even pace across the day.
- Also output "pace": a multiplier on the even schedule (1.0 = exactly on
  schedule, 0.5 = half speed, 2.0 = double speed to catch up).
- Deviate from pace 1.0 only for good reason (bad price -> slow down briefly;
  behind schedule or favourable price -> speed up).""",
    "opportunistic": """- You are running an OPPORTUNISTIC strategy: keep a low baseline participation
  and trade in bursts when the price is favourable vs arrival/VWAP.
- When the price is clearly favourable, ramp target_pov toward the cap with
  "aggressive"; when unfavourable, drop toward zero and wait.""",
}

_COMPLETION_GUIDANCE = {
    True: "- The order MUST complete by end of day -- raise pov/pace if behind schedule.",
    False: ("- Completing the full order is DESIRABLE but NOT mandatory -- do not chase\n"
            "  bad prices just to finish."),
}


def build_decision_prompt(state: dict) -> str:
    """Build the market-snapshot prompt asking the LLM for a trading decision.

    state carries order/market numbers plus an optional "spec" (StrategySpec)
    with the client's compiled natural-language mandate.
    """
    pct_done = 100.0 * state["filled_qty"] / state["total_qty"]
    pct_day_left = 100.0 * (1 - state["bar_index"] / state["n_bars"])
    px_vs_arrival = 100.0 * (state["last_price"] / state["arrival_price"] - 1)

    recent = "\n".join(
        f"  bar -{len(state['recent_bars']) - i}: close={c:.2f}, volume={v:,}"
        for i, (c, v) in enumerate(state["recent_bars"])
    )

    spec = state.get("spec")
    strategy = spec.strategy if spec else "pov"
    must_complete = spec.must_complete if spec else True

    mandate_section = ""
    if spec is not None and spec.mandate_text:
        limit = f"{spec.limit_price:.2f}" if spec.limit_price else "none"
        mandate_section = f"""
CLIENT MANDATE (verbatim): \"{spec.mandate_text}\"
- Compiled strategy: {strategy.upper()} | urgency: {spec.urgency} | hard limit price: {limit}
- Client style instruction: {spec.style_notes or '(none)'}
"""

    schema = ('{"target_pov": <number 0.0-%.2f>, "aggressiveness": "passive" | '
              '"neutral" | "aggressive", "reasoning": "<one short sentence>"}'
              % state["max_pov"])
    if strategy == "twap":
        schema = ('{"pace": <number 0.0-2.0>, "target_pov": <number 0.0-%.2f>, '
                  '"aggressiveness": "passive" | "neutral" | "aggressive", '
                  '"reasoning": "<one short sentence>"}' % state["max_pov"])

    return f"""You are an execution trading agent working a client order.
Your job: trade the order according to the client's mandate while minimising
slippage versus VWAP.
{mandate_section}
ORDER
- Side: {state['side'].upper()}
- Total quantity: {state['total_qty']:,} shares
- Filled so far: {state['filled_qty']:,} shares ({pct_done:.1f}%)
- Remaining: {state['remaining_qty']:,} shares
- Hard participation cap: {state['max_pov']:.2f} (target_pov must be between 0.0 and {state['max_pov']:.2f})
- Realized participation so far: {state['realized_pov']:.3f}
- MINIMUM average POV needed from now on to complete on time: {state['required_pov']:.3f}
  (assumes average volume for the rest of the day -- if this is close to the cap
  you MUST run at high pov with "aggressive" or the order will NOT complete)

MARKET (1-minute bars, most recent last)
- Bar {state['bar_index']} of {state['n_bars']} ({pct_day_left:.0f}% of the day remaining)
- Arrival price: {state['arrival_price']:.2f}
- Last price: {state['last_price']:.2f} ({px_vs_arrival:+.2f}% vs arrival)
- Running market VWAP: {state['market_vwap']:.2f}
- Recent volume per bar: {state['avg_recent_volume']:,.0f} (day average: {state['avg_day_volume']:,.0f})
- Recent bars:
{recent}

GUIDANCE
{_STRATEGY_GUIDANCE[strategy]}
- Execution capture is imperfect: "passive" fills only ~60% of target,
  "neutral" ~90%, "aggressive" 100%. Your EFFECTIVE participation is
  target x capture rate -- account for this shortfall when pacing.
{_COMPLETION_GUIDANCE[must_complete]}

Respond ONLY with a JSON object:
{schema}"""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (e.g. ```json ... ```) some models add."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def parse_decision(text: str, max_pov: float = config.MAX_POV_CAP) -> Decision:
    """Parse the LLM's JSON decision, with a regex fallback and hard clamping."""
    target_pov = None
    aggressiveness = None
    reasoning = ""
    pace = None
    text = _strip_code_fences(text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            target_pov = data.get("target_pov")
            aggressiveness = data.get("aggressiveness")
            reasoning = str(data.get("reasoning", "")).strip()
            pace = data.get("pace")
    except (json.JSONDecodeError, TypeError):
        # Fallback: scrape values out of malformed output.
        pov_match = re.search(r'"?target_pov"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if pov_match:
            target_pov = float(pov_match.group(1))
        pace_match = re.search(r'"?pace"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if pace_match:
            pace = float(pace_match.group(1))
        agg_match = re.search(r"\b(passive|neutral|aggressive)\b", text, re.IGNORECASE)
        if agg_match:
            aggressiveness = agg_match.group(1).lower()
        reason_match = re.search(r'"?reasoning"?\s*[:=]\s*"([^"]+)"', text)
        if reason_match:
            reasoning = reason_match.group(1).strip()

    if not isinstance(target_pov, (int, float)) and not isinstance(pace, (int, float)):
        return neutral_default()

    # Hard guardrails -- the engine never trusts the LLM blindly.
    if isinstance(target_pov, (int, float)):
        target_pov = max(config.MIN_POV, min(float(target_pov), max_pov))
    else:
        target_pov = config.DEFAULT_POV
    if isinstance(pace, (int, float)):
        pace = max(0.0, min(float(pace), 2.0))
    else:
        pace = 1.0
    if aggressiveness not in AGGRESSIVENESS_LEVELS:
        aggressiveness = "neutral"
    if not reasoning:
        reasoning = "(no reasoning given)"

    return Decision(target_pov, aggressiveness, reasoning, pace)
