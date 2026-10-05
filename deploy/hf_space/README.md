---
title: Moneyball Scout
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
short_description: Undervalued football players, explained with SHAP
---

# Moneyball Scout

Predicts the market value of ~4,900 players in 14 European leagues (2025/26), ranks the
most undervalued ones and explains every prediction with SHAP. Includes an honest
backtest of whether flagged players' values actually rose.

Data: Transfermarkt via the Kaggle dataset `davidcariboo/player-scores` (CC0 1.0). Player
photos are not used. Market values are Transfermarkt opinions, not transfer fees.

Optional AI scouting notes need a `GEMINI_API_KEY` Space secret; without it the feature is
hidden. Source code and full methodology: see the GitHub repository.
