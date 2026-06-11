"""All configuration via environment variables (GitHub Secrets in production)."""
import os

# --- secrets ---------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ODDSPAPI_KEY = os.environ.get("ODDSPAPI_KEY", "")

# --- model -----------------------------------------------------------------
MODEL = os.environ.get("MODEL", "claude-opus-4-8")
MAX_LAMBDA_ADJ = float(os.environ.get("MAX_LAMBDA_ADJ", "0.15"))   # hard cap on LLM nudge (+-15%)

# --- odds ------------------------------------------------------------------
# Sharp / tight books to anchor the fair line on (consensus across those present).
SHARP_BOOKS = os.environ.get(
    "SHARP_BOOKS", "pinnacle,polymarket,kalshi,bet365,draftkings"
).split(",")
ODDSPAPI_BASE = "https://api.oddspapi.io/v4"

# --- behaviour -------------------------------------------------------------
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"     # skip LLM + Telegram, just print
STATE_DIR = os.environ.get("STATE_DIR", "state")
SENT_FILE = os.path.join(STATE_DIR, "sent.json")
PRED_LOG = os.path.join(STATE_DIR, "predictions.jsonl")
RESULTS_LOG = os.path.join(STATE_DIR, "results.jsonl")

# Optional static Elo table {team_name: rating} for the cross-check / baseline.
ELO_FILE = os.path.join(STATE_DIR, "elo.json")


def require(*names: str) -> None:
    missing = [n for n in names if not globals().get(n)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
