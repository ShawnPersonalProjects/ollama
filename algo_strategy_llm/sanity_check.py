"""Offline sanity checks (no Ollama required): parsing fallbacks and engine guardrails.

Run: python -m pov_strategy.sanity_check
"""

from .execution import POVExecutor
from .llm_trader import Decision, parse_decision
from .market_data import generate_synthetic_day


def main() -> None:
    # 1. Garbage input -> neutral default
    d = parse_decision("not json at all")
    print("garbage ->", d)
    assert d.target_pov == 0.10 and d.aggressiveness == "neutral"

    # 2. Clamping: LLM says 0.9 -> clamped to cap
    d = parse_decision('{"target_pov": 0.9, "aggressiveness": "aggressive", "reasoning": "go fast"}')
    print("clamp ->", d)
    assert d.target_pov == 0.25

    # 3. Regex fallback on malformed JSON
    d = parse_decision("target_pov: 0.15, be passive please")
    print("regex fallback ->", d)
    assert d.target_pov == 0.15 and d.aggressiveness == "passive"

    # 4. Markdown-fenced JSON (as returned by gemma4:cloud)
    fenced = '```json\n{"target_pov": 0.12, "aggressiveness": "passive", "reasoning": "price below VWAP"}\n```'
    d = parse_decision(fenced)
    print("fenced ->", d)
    assert d.target_pov == 0.12 and d.reasoning == "price below VWAP"

    # 5. Engine never overfills and respects the POV cap per bar
    bars = generate_synthetic_day(seed=42)
    ex = POVExecutor(50_000, "buy", 0.25)
    big = Decision(0.9, "aggressive", "test")
    for i, b in enumerate(bars):
        f = ex.execute_bar(i, b, big)
        if f:
            assert f.qty <= 0.25 * b.volume + 1, f"cap breached at bar {i}"
    assert ex.filled_qty == 50_000, ex.filled_qty
    print(f"engine: filled {ex.filled_qty:,} in {len(ex.fills)} fills, "
          f"realized pov {ex.realized_pov:.3f}")

    # 6. Must-complete failsafe: passive LLM decision gets overridden when
    #    finishing requires participation at/above the cap.
    ex = POVExecutor(2_000_000, "buy", 0.25)
    lazy = Decision(0.02, "passive", "test: far too slow")
    avg_vol = sum(b.volume for b in bars) / len(bars)
    engaged = False
    for i, b in enumerate(bars):
        est = avg_vol * (len(bars) - i)
        f = ex.execute_bar(i, b, lazy, est_future_volume=est)
        if ex.failsafe_engaged and not engaged:
            engaged = True
            assert f is not None and f.aggressiveness == "aggressive"
            assert abs(f.qty / f.bar_volume - 0.25) < 0.01, f.qty / f.bar_volume
    assert engaged, "failsafe never engaged"
    print(f"failsafe: engaged, filled {ex.filled_qty:,} of {ex.total_qty:,} "
          f"({100 * ex.filled_qty / ex.total_qty:.1f}%) at cap+aggressive")

    # 7. Mandate spec validation: garbage/hallucinated compile -> safe defaults
    from .mandate import spec_from_dict
    spec = spec_from_dict({"strategy": "yolo", "side": "short", "total_qty": -5,
                           "max_pov": 3.0, "limit_price": "cheap",
                           "urgency": "extreme"}, "test mandate")
    assert spec.strategy == "pov" and spec.side == "buy"
    assert spec.total_qty == 50_000 and spec.max_pov == 0.50
    assert spec.limit_price is None and spec.must_complete is True
    print("spec validation ->", spec.describe().splitlines()[0])

    # 7b. Amendment keeps unspecified fields from the current spec
    from .mandate import StrategySpec
    current = StrategySpec(strategy="opportunistic", side="sell", total_qty=600_000,
                           max_pov=0.25, limit_price=99.0, must_complete=False)
    amended = spec_from_dict({"urgency": "high", "must_complete": True},
                             "finish it today", base=current)
    assert amended.total_qty == 600_000 and amended.strategy == "opportunistic"
    assert amended.limit_price == 99.0 and amended.must_complete is True
    assert amended.urgency == "high" and amended.side == "sell"
    print("amendment ->", amended.describe().splitlines()[0])

    # 8. Limit price guardrail: buy limit below every close -> zero fills
    ex = POVExecutor(50_000, "buy", 0.25, limit_price=1.0)
    for i, b in enumerate(bars):
        ex.execute_bar(i, b, Decision(0.25, "aggressive", "test"))
    assert ex.filled_qty == 0 and ex.limit_blocked_bars == len(bars)
    print(f"limit price: 0 fills, {ex.limit_blocked_bars} bars blocked")

    # 9. TWAP pacing: pace=1.0 spreads fills roughly evenly across the day
    ex = POVExecutor(100_000, "buy", 0.25, strategy="twap", n_bars=len(bars))
    steady = Decision(0.10, "neutral", "test", pace=1.0)
    for i, b in enumerate(bars):
        ex.execute_bar(i, b, steady)
    first_half = sum(f.qty for f in ex.fills if f.bar_index < len(bars) // 2)
    assert ex.filled_qty > 95_000, ex.filled_qty
    assert 0.35 < first_half / ex.filled_qty < 0.65, first_half / ex.filled_qty
    print(f"twap: filled {ex.filled_qty:,}, first-half share "
          f"{first_half / ex.filled_qty:.2f}")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
