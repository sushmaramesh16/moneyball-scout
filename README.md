# ⚽ Moneyball Scout

**Predict football players' market value, find the undervalued ones, and explain every
prediction, with a backtest that checks whether "undervalued" players' values actually
rose afterwards.**

**Live demo:** _coming soon: `https://<your-app-name>.streamlit.app`_

| Scout a player | Undervalued Gems | Backtest |
|---|---|---|
| _screenshot placeholder: `docs/screenshots/scout.png`_ | _screenshot placeholder: `docs/screenshots/gems.png`_ | _screenshot placeholder: `docs/screenshots/backtest.png`_ |

---

## The problem

Transfermarkt publishes a market value for every professional footballer. Those values are
crowd-and-editor opinions: they lag behind performance, favour famous clubs and react
slowly to new information. Moneyball Scout asks two questions:

1. **What *should* a player be worth**, given his age, position, playing time, output, club
   and league, but *not* his own price history?
2. When the model thinks a player is worth much more than his current value, **does the
   market catch up?** The answer is checked with an out-of-sample backtest, not assumed.

## Data

- **Source:** [Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)
  (`davidcariboo/player-scores`): players, appearances, valuations, games, clubs,
  competitions, transfers and lineups, about 900 MB of CSVs.
- **License:** [CC0 1.0 (public domain)](https://creativecommons.org/publicdomain/zero/1.0/).
  Player photos (`image_url`) are not part of that grant and are **not used**: the app
  shows an initials avatar instead.
- **Scope:** the **14 European leagues** with full player-level appearance data (England,
  Spain, Italy, Germany, France, Netherlands, Portugal, Belgium, Türkiye, Russia, Ukraine,
  Greece, Denmark, Scotland), seasons 2015/16 to 2025/26.
- The raw data is never committed (`data/` is gitignored). The deployed app runs on two
  small artifacts (a model and a player table, about 11 MB in total).

## Approach

### 1. A point-in-time player-season panel
One row = one player in one season (**53,291 rows, 16,066 players**), built with DuckDB
(`src/features/build_features.py`, about 4 s). Every feature uses only data dated **on or
before that league's final matchday**. Snapshot fields that can't be reconstructed for
past seasons are excluded: contract length, international caps and the clubs table's
current squad stats. They are rebuilt per season where possible: club strength, squad
age and transfer activity from games, appearances and transfers; position from that
season's lineups.

Features cover age, position, minutes and output (this season, last season and the last
three), share of the club's minutes, European and cup football, club league position and
goal difference, club transfer spending (excluding the player's own fee), and league
price level. The target is the player's valuation nearest the season's end.

**Never used by the main model:** the player's own past valuations or transfer fees.
Otherwise the model would just echo the market, and "undervalued" would mean "his value
dropped".

### 2. Leakage tests
- **Time-travel test:** the panel is rebuilt from the raw data truncated to 30 June 2024.
  All **43,629** player-seasons up to 2023/24 come out with **identical values for all 58
  features**.
- **Synthetic tests:** matches, transfers and valuations added *after* a season's end must
  not change that season's features, and a control test shows that data added *before*
  season end does change them.
- **Validator:** a per-row point-in-time audit runs on every build.

### 3. A market index instead of raw euros
Football prices inflate (about +60% over the decade), so a model trained on old seasons
underprices new ones. The target is **log value relative to a market index**: the mean
log value, as of 1 July, of players who had 450+ minutes in the 14 leagues the season
before. A fixed population matters, because recent seasons have far fewer valuations for
fringe players, which would look like inflation. League indexes enter as features.

### 4. Models
LightGBM (native categoricals, tuned with Optuna on 2023/24) against a linear baseline,
Random Forest and XGBoost, with every run tracked in MLflow. The time split is train
2015/16–2022/23, validate 2023/24, **test 2024/25 (used once)**, then retrain through
2024/25 to score the live 2025/26 season. A separate *ceiling* model adds valuation
history, for reference only.

### 5. Undervalued score and explanations
- **Raw score:** (predicted − actual) / actual.
- **De-biased score (default):** the same gap compared with peers in the same
  league × age band. The model systematically over-values veterans and some leagues; this
  removes that bias.
- **SHAP (TreeSHAP):** explains every prediction. The app shows each player's drivers in
  plain terms, flags when a case rests mostly on *context* (age, club, league) rather than
  *performance*, and shows a value range from the test-set error spread.

## Results

### Valuation accuracy: held-out test season 2024/25

| Model | RMSE (log) | R² | Median abs. error | Typical miss |
|---|---|---|---|---|
| **LightGBM (main model)** | **0.458** | **0.911** | **€635K** | **34%** |
| Linear baseline | 0.562 | 0.865 | €739K | 45% |
| *Reference: with past valuations (ceiling)* | 0.194 | 0.984 | €220K | 11% |
| *Reference: last valuation carried forward* | 0.267 | 0.970 | €150K | 14% |

The reference models use the player's own price history, so they mostly reproduce the
market. The gap between them and the main model is where undervalued signals can live,
and also where noise lives.

### Backtest: do flagged players' values rise? (headline: 2024/25, out-of-sample, no value floor)

At the end of 2024/25 every player is ranked by a model trained only on earlier seasons.
"Hit" = Transfermarkt value higher about 12 months later. Brackets are 95% bootstrap
intervals (2,000 resamples of players, re-selecting the top group each time).

| Strategy (2024/25) | Hit rate | Lift vs all players | Matched lift (age × value) |
|---|---|---|---|
| All players (base rate) | 30.4% | 1.00 | 1.00 |
| **Model, de-biased, top 10%** | **36.5% [31–41%]** | **1.20 [1.04–1.36]** | **1.16 [1.02–1.28]** |
| Model, de-biased, top 100 | 43.8% [32–55%] | 1.44 [1.04–1.82] | 1.35 [1.03–1.64] |
| Model, raw, top 10% | 32.3% [27–36%] | 1.06 [0.90–1.19] | 1.21 [1.06–1.34] |
| Model, raw, top 100 | 25.7% [16–38%] | 0.84 [0.54–1.25] | 1.15 [0.82–1.62] |
| Linear model, top 10% | 27.6% | 0.91 | 1.15 |
| Young regulars (U23 by minutes), top 10% | 53.9% [49–58%] | 1.77 [1.62–1.91] | 1.10 [1.02–1.17] |
| Mean reversion (biggest recent drop), top 10% | 11.8% [9–16%] | 0.39 [0.30–0.52] | 0.71 [0.55–0.91] |

Rank correlation between score and 12-month value change across all players: de-biased
**0.094** [0.063–0.125], raw 0.029 [−0.003–0.060], linear 0.033.

**What is "matched lift"?** Young and cheap players rise more often whatever any model
says (the age curve, and small values moving up more easily). So for each flagged group
we compute the hit rate expected from its **age × value-band mix alone**, using every
player's outcome that season. Matched lift = actual hit rate ÷ that expectation; 1.0
means the score adds nothing beyond age and price. That's why "young regulars" look great
on raw hit rate (54%) but add little once matched (1.10), while the de-biased model's
picks beat their matched baseline by about 16% (top 10%) to 35% (top 100).

Other checks:
- **Bounce-back fails.** Players whose value just dropped keep falling (11.8% rise).
- **2023/24** shows the same ordering (de-biased top-10% hit 38.9%, rank correlation
  0.110) but is labelled optimistic, because hyperparameters were tuned on it.
- **Value floor.** The app's default €500K floor was chosen after looking at 2024/25 and
  **validated on 2023/24**, where it made no material difference (top-10% hit 38.9% →
  39.1%). It is a usability choice: ratio scores explode for near-zero valuations.

## Limitations

- **Transfermarkt values are opinions, not transfer fees.** The model predicts what
  Transfermarkt *would* value a player at, and the backtest checks whether *Transfermarkt*
  moved. Neither says what a club would pay.
- **14 leagues only.** Players elsewhere have no appearance data; moves out of these
  leagues are partly invisible.
- **Missing outcomes.** About 18% of players have no valuation 12 months later, and
  flagged players are missing more often (24.7% vs 17.3%, p < 0.001), mostly because cheap
  players are re-valued less often. Counting every missing outcome as "did not rise", the
  de-biased top 10% still has a matched lift of 1.14.
- **The lift is modest.** About 16% over the age × value baseline (CI 2–28%) for the top
  10% in one out-of-sample season. That's a useful filter for scouts, not a money machine.
- **Many flags are context-driven.** About 7 in 10 of the top-100 gems are flagged mainly
  because of age, club or league rather than output. The app says so per player.
- **No tactics, injuries, contracts or video.** Contract length exists only as a current
  snapshot (shown for context, never used by the model).
- **Typical error is about 34%**, so the app shows a value range, not just a point
  estimate.
- **AI scouting notes** (optional, Gemini) are written only from the numbers shown and are
  labelled AI-generated; they can still be wrong.

## How to run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # everything (app runtime + training + dev)

# Use the committed artifacts: no raw data needed
streamlit run app/streamlit_app.py         # http://localhost:8501
uvicorn src.api.main:app --reload          # http://localhost:8000/docs

# Or both with Docker
docker compose up --build                  # API :8000, Streamlit :8501 (via the API)

# Optional AI scouting notes
cp .env.example .env                       # then set GEMINI_API_KEY
```

### Rebuild everything from the raw data
Download the Kaggle dataset into `data/raw/` (12 CSVs), then:

```bash
python -m src.features.build_features      # point-in-time panel -> data/processed/
python -m src.models.train compare         # model comparison (MLflow)
python -m src.models.train ablate          # feature decisions
python -m src.models.tune --trials 60      # Optuna on 2023/24
python -m src.models.final test            # 2024/25 test (refuses to run twice)
python -m src.models.backtest              # backtests, bootstrap CIs, value floor
python -m src.models.final intervals       # value ranges from test residuals
python -m src.models.final live            # 2025/26 scores + deployment artifacts
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### API

| Endpoint | Description |
|---|---|
| `GET /undervalued` | Ranked list; filters `position_group`, `position`, `league`, `min_age`, `max_age`, `min_value` (default €500K), `score=debiased\|raw` |
| `GET /explain/{player_id}` | Full SHAP breakdown, plain-language values, context vs performance |
| `POST /predict` | Score any feature vector (missing features allowed) |
| `GET /report/{player_id}` | AI scouting note (503 without `GEMINI_API_KEY`, 429 when rate-limited) |
| `GET /players`, `/players/{id}`, `/backtest`, `/model/importance`, `/health` | Search, details, backtest tables, global SHAP |

### Tests & CI
`pytest` runs 138 tests covering cleaning, feature values, leakage, models, the backtest
math, SHAP additivity, the API, the Streamlit pages and the report logic (with a fake
LLM). Real-data leakage tests skip automatically when `data/raw` is absent, as in CI.
GitHub Actions runs `ruff check`, `ruff format --check`, `pytest` and a Docker build with
an API smoke test.

## Deploy (Streamlit Community Cloud, free)

The live demo runs on [Streamlit Community Cloud](https://streamlit.io/cloud), straight
from this GitHub repo. No API server is needed: the app loads `artifacts/` directly.

- **Entrypoint:** `app/streamlit_app.py`, with **Python 3.11** chosen under *Advanced
  settings*.
- **Dependencies:** `app/requirements.txt` sits next to the entrypoint, so Community Cloud
  installs it instead of the root `requirements.txt`. It holds only the pinned runtime
  (no mlflow, xgboost or duckdb), with versions kept identical to
  `requirements-deploy.txt` by a test.
- **System packages:** `packages.txt` installs `libgomp1` (OpenMP, needed by LightGBM).
- **AI notes (optional):** add `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) in the
  app's *Secrets* as TOML. The app reads environment variables first, then
  `st.secrets`. Locally you can use `.env` or `.streamlit/secrets.toml` (both gitignored).
  Without a key, the AI section is hidden.
- **Report cache:** stored in `.cache/reports`, falling back to the system temp dir if
  that isn't writable.

**Docker** remains the way to run the full stack (FastAPI + Streamlit) locally:
`docker compose up --build`. The same image also works as a Hugging Face **Docker
Space** (`./scripts/build_hf_space.sh` assembles the bundle), but Docker Spaces now
require a paid Hugging Face plan, so the free demo uses Community Cloud instead.

## Project layout

```
src/data/        load (typed DuckDB views, as_of truncation), clean, validate
src/features/    build_features (point-in-time panel), feature_sets
src/models/      pipelines, train (compare/ablate), tune, final, backtest, undervalued, evaluate
src/explain/     shap_explain (SHAP + plain-language values), report (Gemini)
src/service.py   artifact-backed service shared by the API and the app
src/api/         FastAPI app + schemas
app/             Streamlit app (scout, gems, backtest, model pages)
artifacts/       model.joblib, players_live.parquet, intervals, SHAP importance
reports/         comparison, tuning, test and backtest tables
tests/           138 tests, including synthetic Transfermarkt-shaped fixtures
```
