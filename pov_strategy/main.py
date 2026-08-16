"""Run an LLM-driven execution strategy defined in natural language.

Usage:
    # Classic flags (no mandate compilation):
    python -m pov_strategy.main --qty 50000 --side buy

    # Natural-language mandate -> compiled strategy:
    python -m pov_strategy.main --mandate "Sell 750k shares today, stay under
        15% of volume, never below 99.50, ok to leave some unfilled"

    # Preset mandate gallery:
    python -m pov_strategy.main --preset urgent

    # Interactive: type mandates, get runs and reports in a loop:
    python -m pov_strategy.main --repl
"""

import argparse

from . import config
from .execution import POVExecutor
from .llm_trader import build_decision_prompt, neutral_default, parse_decision
from .mandate import StrategySpec, compile_mandate
from .market_data import Bar, generate_synthetic_day, load_yfinance
from .ollama_client import ask_model
from .report import compute_metrics, print_summary, save_charts

PRESETS = {
    "conservative": (
        "Buy 400,000 shares over the day. Keep a low profile: never more than "
        "10% of volume, patience over speed, and prefer buying below VWAP. "
        "It is acceptable to leave a small portion unfilled."
    ),
    "urgent": (
        "I need to buy 1,000,000 shares today no matter what. Prioritise "
        "certainty of completion over price. Cap participation at 25%."
    ),
    "opportunistic": (
        "Sell 600,000 shares opportunistically: sit quietly and unload in "
        "bursts when the price pops above VWAP. Never sell below 99.00. "
        "Finishing the order is nice to have, not a requirement."
    ),
    "steady": (
        "Buy 500,000 shares at an even pace across the whole day (TWAP style), "
        "no more than 20% of volume at any time. Must be done by the close."
    ),
}


def build_state(bars: list[Bar], i: int, executor: POVExecutor,
                avg_day_volume: float, spec: StrategySpec) -> dict:
    lookback = bars[max(0, i - config.LOOKBACK_BARS):i + 1]
    est_future_volume = avg_day_volume * (len(bars) - i)
    required_pov = executor.remaining_qty / max(1.0, est_future_volume)
    return {
        "side": executor.side,
        "total_qty": executor.total_qty,
        "filled_qty": executor.filled_qty,
        "remaining_qty": executor.remaining_qty,
        "max_pov": executor.max_pov,
        "bar_index": i,
        "n_bars": len(bars),
        "arrival_price": executor.arrival_price or bars[0].open,
        "last_price": bars[i].close,
        "market_vwap": executor.market_vwap or bars[i].close,
        "recent_bars": [(b.close, b.volume) for b in lookback],
        "avg_recent_volume": sum(b.volume for b in lookback) / len(lookback),
        "avg_day_volume": avg_day_volume,
        "realized_pov": executor.realized_pov,
        "required_pov": min(required_pov, 1.0),
        "spec": spec,
    }


def load_bars(args: argparse.Namespace) -> list[Bar]:
    if args.source == "synthetic":
        bars = generate_synthetic_day(seed=args.seed)
        print(f"Generated {len(bars)} synthetic 1-minute bars.")
    else:
        bars = load_yfinance(args.symbol)
        print(f"Loaded {len(bars)} bars for {args.symbol} from yfinance.")
    return bars


def read_mandate_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def apply_amendment(executor: POVExecutor, old_spec: StrategySpec,
                    new_spec: StrategySpec) -> StrategySpec:
    """Apply an amended spec to the in-flight order (side cannot change)."""
    if new_spec.side != old_spec.side:
        print(f"WARNING: side change ({old_spec.side} -> {new_spec.side}) "
              "ignored mid-run.")
        new_spec.side = old_spec.side
    new_spec.total_qty = max(new_spec.total_qty, executor.filled_qty)
    executor.total_qty = new_spec.total_qty
    executor.max_pov = new_spec.max_pov
    executor.limit_price = new_spec.limit_price
    executor.must_complete = new_spec.must_complete
    executor.strategy = new_spec.strategy
    executor.failsafe_engaged = False  # re-evaluate under the new spec
    return new_spec


def run_spec(spec: StrategySpec, args: argparse.Namespace,
             mandate_file: str | None = None) -> None:
    bars = load_bars(args)
    executor = POVExecutor(
        spec.total_qty, spec.side, spec.max_pov,
        limit_price=spec.limit_price, must_complete=spec.must_complete,
        strategy=spec.strategy, n_bars=len(bars),
    )
    decision = neutral_default("initial decision (LLM not yet consulted)")
    avg_day_volume = sum(b.volume for b in bars) / len(bars)
    failsafe_announced = False
    last_mandate_text = spec.mandate_text

    print(f"\nCOMPILED SPEC: {spec.describe()}")
    print(f"model={args.model}, deciding every {args.decide_every} bars\n")

    for i, bar in enumerate(bars):
        if executor.done:
            break

        if i % args.decide_every == 0:
            # Live amendment: the human trader can rewrite the mandate file
            # mid-run; the very next decision trades under the new mandate.
            if mandate_file:
                text = read_mandate_file(mandate_file)
                if text and text != last_mandate_text:
                    print(f"\n*** MANDATE AMENDED at {bar.ts:%H:%M} (bar {i}) ***")
                    print(f'NEW MANDATE: "{text}"')
                    try:
                        new_spec = compile_mandate(text, model=args.model,
                                                   current=spec)
                        spec = apply_amendment(executor, spec, new_spec)
                        failsafe_announced = False
                        print(f"UPDATED SPEC: {spec.describe()}\n")
                    except RuntimeError as e:
                        print(f"Amendment compile failed, keeping old spec: {e}\n")
                    last_mandate_text = text

            state = build_state(bars, i, executor, avg_day_volume, spec)
            prompt = build_decision_prompt(state)
            if args.verbose:
                print("\n" + "-" * 70)
                print(f"PROMPT (bar {i}, {bar.ts:%H:%M}):")
                print("-" * 70)
                print(prompt)
            try:
                raw = ask_model(prompt, model=args.model)
                if args.verbose:
                    print("-" * 70)
                    print("RAW MODEL RESPONSE:")
                    print(raw)
                    print("-" * 70)
                decision = parse_decision(raw, spec.max_pov)
            except RuntimeError as e:
                decision = neutral_default(f"fallback: {e}")
            pace_info = f" pace={decision.pace:.2f}" if spec.strategy == "twap" else ""
            print(f"[bar {i:3d} | {bar.ts:%H:%M} | filled {executor.filled_qty:,}/"
                  f"{executor.total_qty:,}] pov={decision.target_pov:.3f}"
                  f"{pace_info} ({decision.aggressiveness}) -- {decision.reasoning}")

        executor.execute_bar(i, bar, decision,
                             est_future_volume=avg_day_volume * (len(bars) - i))
        if executor.failsafe_engaged and not failsafe_announced:
            print(f"[bar {i:3d} | {bar.ts:%H:%M}] *** MUST-COMPLETE FAILSAFE ENGAGED: "
                  f"engine overriding LLM with cap + aggressive ***")
            failsafe_announced = True

    metrics = compute_metrics(executor, bars)
    print_summary(metrics)
    chart_path = save_charts(executor, bars, metrics, path=args.chart)
    print(f"Charts saved to {chart_path}")


def spec_from_args(args: argparse.Namespace) -> StrategySpec:
    """Build a spec directly from CLI flags (no LLM compilation)."""
    return StrategySpec(side=args.side, total_qty=args.qty, max_pov=args.max_pov)


def compile_and_show(mandate: str, model: str) -> StrategySpec:
    print(f'\nMANDATE: "{mandate}"')
    print("Compiling mandate with LLM...")
    return compile_mandate(mandate, model=model)


def repl(args: argparse.Namespace) -> None:
    print("Natural-language strategy REPL. Describe an execution strategy in "
          "plain English.\nExamples:")
    for name, text in PRESETS.items():
        print(f'  [{name}] "{text[:70]}..."')
    print("Empty line to exit.\n")
    run_no = 0
    while True:
        try:
            mandate = input("mandate> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not mandate:
            break
        run_no += 1
        args.chart = f"pov_report_repl_{run_no}.png"
        spec = compile_and_show(mandate, args.model)
        run_spec(spec, args)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-driven execution strategies from natural language")
    parser.add_argument("--mandate", help="plain-English strategy mandate to compile and run")
    parser.add_argument("--mandate-file",
                        help="file holding the mandate; edit it mid-run to amend "
                             "the strategy live")
    parser.add_argument("--preset", choices=sorted(PRESETS),
                        help="run a canned example mandate")
    parser.add_argument("--repl", action="store_true",
                        help="interactive mandate loop")
    parser.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    parser.add_argument("--symbol", default="AAPL", help="ticker for --source yfinance")
    parser.add_argument("--qty", type=int, default=config.DEFAULT_QTY,
                        help="used only without --mandate/--preset/--repl")
    parser.add_argument("--side", choices=["buy", "sell"], default=config.DEFAULT_SIDE)
    parser.add_argument("--max-pov", type=float, default=config.MAX_POV_CAP)
    parser.add_argument("--decide-every", type=int, default=config.DECIDE_EVERY,
                        help="consult the LLM every N bars")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--seed", type=int, default=None, help="synthetic data seed")
    parser.add_argument("--chart", default="pov_report.png", help="output chart path")
    parser.add_argument("--verbose", action="store_true",
                        help="print each full prompt and raw LLM response")
    args = parser.parse_args()

    if not 0 < args.max_pov <= 1:
        parser.error("--max-pov must be in (0, 1]")
    if args.qty <= 0:
        parser.error("--qty must be positive")
    if args.decide_every <= 0:
        parser.error("--decide-every must be positive")

    if args.repl:
        repl(args)
    elif args.mandate_file:
        mandate = read_mandate_file(args.mandate_file)
        if not mandate:
            parser.error(f"could not read a mandate from {args.mandate_file}")
        spec = compile_and_show(mandate, args.model)
        run_spec(spec, args, mandate_file=args.mandate_file)
    elif args.mandate or args.preset:
        mandate = args.mandate or PRESETS[args.preset]
        spec = compile_and_show(mandate, args.model)
        run_spec(spec, args)
    else:
        run_spec(spec_from_args(args), args)


if __name__ == "__main__":
    main()
