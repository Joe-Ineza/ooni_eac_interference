"""
Central configuration for OONI EAC Social Media Throttling Analysis.
Adjust parameters here without touching any analysis logic.
"""

# ── DATE RANGE ─────────────────────────────────────────────────────────────────
# Change START_DATE to expand the analysis window (e.g., "2020-01-01" for full study)
START_DATE = "2024-01-01"
END_DATE   = "2026-01-31"   # Inclusive upper bound for data collection

# ── TARGET COUNTRIES ────────────────────────────────────────────────────────────
COUNTRIES = {
    "KE": "Kenya",
    "UG": "Uganda",
    "TZ": "Tanzania",
}

# Political regime classification for RQ1 comparative analysis.
# Source: Freedom House + V-Dem scores (2024). Update as needed.
REGIME_TYPE = {
    "KE": "competitive_authoritarian",   # Partly free; competitive but constrained
    "UG": "authoritarian",               # Not free; consolidated executive power
    "TZ": "hybrid",                      # Partly free; democratic regression post-2015
}


# ── PLATFORMS / TEST TARGETS ────────────────────────────────────────────────────
# Canonical domains queried via web_connectivity (one per platform, no www. variants).
# Twitter/X has no dedicated OONI test so it is covered here.
# Facebook and WhatsApp are covered by dedicated tests below -- not listed here
# to avoid double-counting. Telegram likewise.
SOCIAL_MEDIA_DOMAINS = [
    "twitter.com",      # X / Twitter  (no dedicated OONI test)
    "tiktok.com",       # TikTok       (no dedicated OONI test)
    "youtube.com",      # YouTube      (no dedicated OONI test)
    "instagram.com",    # Instagram    (no dedicated OONI test)
]

# OONI dedicated platform tests.
# signal removed -- low probe coverage in EAC.
# twitter/X NOT listed here -- covered via web_connectivity above.
TEST_NAMES = [
    "web_connectivity",     # twitter, tiktok, youtube, instagram
    "whatsapp",             # dedicated WhatsApp reachability test
    "facebook_messenger",   # dedicated Facebook/Messenger test
    "telegram",             # dedicated Telegram test
]

# Max pages fetched per domain in web_connectivity (1000 records/page).
# Caps at 5000 records per domain -- prevents runaway on high-volume domains.
MAX_PAGES_PER_DOMAIN = 5



# ── ELECTORAL EVENT CALENDAR ────────────────────────────────────────────────────
# Key dates for temporal anchoring. T=0 is election day.
# Format: {"label": str, "date": "YYYY-MM-DD", "country": CC or "ALL"}
ELECTORAL_EVENTS = [
    # Kenya major protest wave (expands pre-2025 pilot coverage)
    {"label": "KE_finance_bill_protests",  "date": "2024-06-25",  "country": "KE"},

    # Uganda general election cycle (run-up to Feb 2026 election)
    {"label": "UG_campaign_period_start", "date": "2025-09-01",  "country": "UG"},
    {"label": "UG_nomination_day",        "date": "2025-11-03",  "country": "UG"},
    {"label": "UG_election_day",          "date": "2026-01-14",  "country": "UG"},

    # Kenya (no major national election in window, but notable protest periods)
    {"label": "KE_anti_govt_protests",    "date": "2025-06-20",  "country": "KE"},

    # Tanzania (local government elections)
    {"label": "TZ_local_elections",       "date": "2025-11-27",  "country": "TZ"},
]

# Days before/after event to include in temporal window analysis
EVENT_WINDOW_DAYS_BEFORE = 14
EVENT_WINDOW_DAYS_AFTER  = 21

# ── RECOVERY DETECTION SETTINGS (RQ2) ─────────────────────────────────────────
# Pilot v1 defaults are intentionally relaxed to increase observed recovery labels.
RECOVERY_VOLUME_BASELINE_RATIO_THRESHOLD = 0.75
RECOVERY_HARD_BLOCK_FRAC_THRESHOLD = 0.10
RECOVERY_MIN_CONSECUTIVE_DAYS = 1

# ── API SETTINGS ────────────────────────────────────────────────────────────────
OONI_API_BASE      = "https://api.ooni.io/api/v1"
OONI_ANALYSIS_EP   = f"{OONI_API_BASE}/analysis"
OONI_MEASUREMENT_EP = f"{OONI_API_BASE}/measurements"

# Polite rate limiting — OONI asks for modest request rates
REQUEST_DELAY_SECONDS = 1.0   # Pause between paginated requests
PAGE_SIZE             = 1000  # Records per API page (max for analysis endpoint)

# ── ANOMALY THRESHOLDS (RQ1) ────────────────────────────────────────────────────
# Proportion of measurements showing a failure mode to flag as anomalous.
# Based on Sundara Raman et al. (2023) recommendations.
DNS_BLOCK_THRESHOLD = 0.10   # 10%+ dns_blocked → likely DNS tampering
TCP_BLOCK_THRESHOLD = 0.10   # 10%+ tcp_blocked → likely IP-level blocking
TLS_BLOCK_THRESHOLD = 0.10   # 10%+ tls_blocked → likely DPI / SNI filtering

# ── OUTPUT PATHS ────────────────────────────────────────────────────────────────
import os
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
RAW_DATA_FILE   = os.path.join(DATA_DIR, "raw_ooni_data.jsonl")
CLEAN_DATA_FILE = os.path.join(DATA_DIR, "clean_ooni_data.parquet")
