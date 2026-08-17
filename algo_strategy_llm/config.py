# --- Ollama ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:cloud"
REQUEST_TIMEOUT = 180  # seconds

# --- Parent order defaults ---
DEFAULT_QTY = 50_000
DEFAULT_SIDE = "buy"  # "buy" or "sell"

# --- Guardrails (enforced by the engine, regardless of LLM output) ---
MAX_POV_CAP = 0.25   # hard ceiling on participation rate
MIN_POV = 0.0
DEFAULT_POV = 0.10   # fallback when the LLM output is unusable

# --- Decision cadence ---
DECIDE_EVERY = 5     # consult the LLM every N bars
LOOKBACK_BARS = 10   # how many recent bars to show the LLM

# --- Synthetic market defaults ---
SYNTHETIC_BARS = 390          # 1-minute bars in a US trading day
SYNTHETIC_START_PRICE = 100.0
SYNTHETIC_DAILY_VOL = 0.02    # ~2% daily volatility
SYNTHETIC_DAY_VOLUME = 5_000_000  # total shares traded across the day
