"""Why the 1M-share order can't finish: compute the fill ceiling for seed 7.

Run: python -m pov_strategy.why_unfilled
"""

from .market_data import generate_synthetic_day

bars = generate_synthetic_day(seed=7)
day_vol = sum(b.volume for b in bars)
print(f"Total day volume:            {day_vol:,}")
print(f"Order size:                  1,000,000 ({100 * 1_000_000 / day_vol:.1f}% of day volume)")
print(f"Ceiling at 25% cap all day (aggressive, fill_ratio=1.0): {0.25 * day_vol:,.0f}")
print(f"Ceiling at 25% cap all day (neutral,    fill_ratio=0.9): {0.25 * 0.9 * day_vol:,.0f}")

# Approximate the actual run: decisions from the verbose log.
schedule = [(0, 0.05, 0.60), (60, 0.12, 0.60), (120, 0.15, 0.90),
            (180, 0.18, 0.90), (240, 0.22, 0.90), (300, 0.25, 0.90)]
filled = 0.0
for start, pov, ratio in schedule:
    end = start + 60 if start < 300 else len(bars)
    vol = sum(b.volume for b in bars[start:end])
    got = pov * ratio * vol
    filled += got
    print(f"bars {start:3d}-{end:3d}: pov={pov:.2f} ratio={ratio:.2f} "
          f"segment_vol={vol:>9,} -> ~{got:>9,.0f} filled (cum ~{filled:,.0f})")

vol_after_240 = sum(b.volume for b in bars[240:])
print(f"\nVolume remaining after 13:30 (bar 240): {vol_after_240:,}")
print(f"Max fillable from 13:30 even at cap+aggressive: {0.25 * vol_after_240:,.0f}")
