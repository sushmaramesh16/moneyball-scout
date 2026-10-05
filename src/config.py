"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
PANEL_PATH = PROCESSED_DIR / "panel.parquet"

# The 14 domestic leagues for which appearances.csv has full player-level coverage.
LEAGUES = (
    "GB1",
    "ES1",
    "IT1",
    "L1",
    "FR1",
    "NL1",
    "PO1",
    "BE1",
    "TR1",
    "RU1",
    "UKR1",
    "GR1",
    "DK1",
    "SC1",
)

# UEFA club competitions. Note the dataset types EL / UECL as "other", not "international_cup".
EURO_COMPS = ("CL", "CLQ", "EL", "ELQ", "UCOL", "ECLQ", "USC")

FIRST_PANEL_SEASON = 2015  # 2012-2014 are only used as lag/history seasons
MIN_MINUTES = 450  # eligibility: minutes in all competitions up to season end
TARGET_WINDOW_DAYS = 120  # target = valuation nearest season end, within +/- this window
INDEX_LOOKBACK_DAYS = 365  # market index: each player's latest valuation within this lookback

# Time-based split (season = starting year, 2023 means 2023/24)
TRAIN_LAST_SEASON = 2022
VALID_SEASON = 2023
TEST_SEASON = 2024
LIVE_SEASON = 2025

# Backtest: "12 months later" valuation, searched within +/- this many days of target + 365
FUTURE_WINDOW_DAYS = 60
