"""Post-run metrics and charts for a POV execution."""

from .execution import POVExecutor
from .market_data import Bar


def compute_metrics(executor: POVExecutor, bars: list[Bar]) -> dict:
    day_volume = sum(b.volume for b in bars)
    day_notional = sum(b.volume * b.close for b in bars)
    day_vwap = day_notional / day_volume if day_volume else 0.0

    avg_fill = executor.avg_fill_price
    arrival = executor.arrival_price or 0.0
    # For buys, paying above the benchmark is a cost; for sells, below.
    sign = 1 if executor.side == "buy" else -1
    slippage_vs_vwap_bps = sign * (avg_fill - day_vwap) / day_vwap * 10_000 if day_vwap else 0.0
    shortfall_bps = sign * (avg_fill - arrival) / arrival * 10_000 if arrival else 0.0

    return {
        "side": executor.side,
        "total_qty": executor.total_qty,
        "filled_qty": executor.filled_qty,
        "completion_pct": 100.0 * executor.filled_qty / executor.total_qty,
        "avg_fill_price": avg_fill,
        "arrival_price": arrival,
        "day_vwap": day_vwap,
        "slippage_vs_vwap_bps": slippage_vs_vwap_bps,
        "implementation_shortfall_bps": shortfall_bps,
        "realized_pov": executor.realized_pov,
        "n_fills": len(executor.fills),
        "limit_blocked_bars": executor.limit_blocked_bars,
    }


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("POV EXECUTION REPORT")
    print("=" * 60)
    print(f"  Side:                     {metrics['side'].upper()}")
    print(f"  Filled:                   {metrics['filled_qty']:,} / {metrics['total_qty']:,} "
          f"({metrics['completion_pct']:.1f}%)")
    print(f"  Child orders (fills):     {metrics['n_fills']}")
    print(f"  Avg fill price:           {metrics['avg_fill_price']:.4f}")
    print(f"  Arrival price:            {metrics['arrival_price']:.4f}")
    print(f"  Day VWAP:                 {metrics['day_vwap']:.4f}")
    print(f"  Slippage vs VWAP:         {metrics['slippage_vs_vwap_bps']:+.2f} bps")
    print(f"  Implementation shortfall: {metrics['implementation_shortfall_bps']:+.2f} bps")
    print(f"  Realized participation:   {metrics['realized_pov'] * 100:.2f}%")
    if metrics["limit_blocked_bars"]:
        print(f"  Bars blocked by limit px: {metrics['limit_blocked_bars']}")
    print("=" * 60)


def save_charts(executor: POVExecutor, bars: list[Bar], metrics: dict,
                path: str = "pov_report.png") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [b.ts for b in bars]
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]

    # Running market VWAP series.
    vwap_series, cum_v, cum_pv = [], 0, 0.0
    for b in bars:
        cum_v += b.volume
        cum_pv += b.volume * b.close
        vwap_series.append(cum_pv / cum_v)

    child_qty = [0] * len(bars)
    pov_per_bar = [0.0] * len(bars)
    target_pov = [0.0] * len(bars)
    for f in executor.fills:
        child_qty[f.bar_index] = f.qty
        pov_per_bar[f.bar_index] = f.qty / f.bar_volume
        target_pov[f.bar_index] = f.target_pov

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"POV Execution -- {metrics['side'].upper()} {metrics['total_qty']:,} shares | "
        f"slippage vs VWAP: {metrics['slippage_vs_vwap_bps']:+.1f} bps",
        fontsize=12,
    )

    ax1.plot(times, closes, label="Price", linewidth=1)
    ax1.plot(times, vwap_series, label="Market VWAP", linestyle="--", linewidth=1)
    fill_times = [f.ts for f in executor.fills]
    fill_prices = [f.price for f in executor.fills]
    ax1.scatter(fill_times, fill_prices, s=8, color="red", label="Fills", zorder=3)
    ax1.axhline(metrics["avg_fill_price"], color="green", linewidth=0.8,
                label=f"Avg fill {metrics['avg_fill_price']:.2f}")
    ax1.set_ylabel("Price")
    ax1.legend(loc="best", fontsize=8)

    ax2.bar(times, volumes, width=0.0006, color="lightgray", label="Market volume")
    ax2.bar(times, child_qty, width=0.0006, color="steelblue", label="Child qty")
    ax2.set_ylabel("Shares")
    ax2.legend(loc="best", fontsize=8)

    ax3.plot(times, [p * 100 for p in pov_per_bar], label="Realized POV", linewidth=1)
    ax3.plot(times, [p * 100 for p in target_pov], label="LLM target POV",
             linestyle=":", linewidth=1)
    ax3.axhline(executor.max_pov * 100, color="red", linewidth=0.8,
                label=f"Cap {executor.max_pov * 100:.0f}%")
    ax3.set_ylabel("POV (%)")
    ax3.set_xlabel("Time")
    ax3.legend(loc="best", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
