"""
World Cup 2026 Predictor — Phase 1: Database Construction & Feature Engineering
==============================================================================
Builds a clean relational database from the Fjelstul World Cup Database
(men's tournaments 1930-2022) and engineers predictive features at the
match level using ONLY information available before each match (no leakage).

Author: Edgar Navarro | Bilingual Data Analyst
Data source: Fjelstul World Cup Database (CC-BY-SA 4.0)
"""
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD & FILTER — men's tournaments only, drop replays to avoid duplicates
# ---------------------------------------------------------------------------
matches = pd.read_csv(DATA / "matches.csv")
matches = matches[matches["tournament_name"].str.contains("Men's")].copy()
matches = matches[matches["replay"] == 0].copy()
matches["match_date"] = pd.to_datetime(matches["match_date"])
matches["year"] = matches["match_date"].dt.year
matches = matches.sort_values("match_date").reset_index(drop=True)

print(f"Loaded {len(matches)} men's WC matches ({matches['year'].min()}-{matches['year'].max()})")

# ---------------------------------------------------------------------------
# 2. BUILD A LONG (TEAM-PER-ROW) VIEW — one row per team per match
#    This makes rolling/historical features far easier to compute correctly.
# ---------------------------------------------------------------------------
def to_long(df):
    home = pd.DataFrame({
        "match_key": df["key_id"], "date": df["match_date"], "year": df["year"],
        "tournament": df["tournament_name"], "stage": df["stage_name"],
        "team": df["home_team_name"], "opponent": df["away_team_name"],
        "goals_for": df["home_team_score"], "goals_against": df["away_team_score"],
        "is_home_listed": 1,
    })
    away = pd.DataFrame({
        "match_key": df["key_id"], "date": df["match_date"], "year": df["year"],
        "tournament": df["tournament_name"], "stage": df["stage_name"],
        "team": df["away_team_name"], "opponent": df["home_team_name"],
        "goals_for": df["away_team_score"], "goals_against": df["home_team_score"],
        "is_home_listed": 0,
    })
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["goals_for"] > long["goals_against"]).astype(int)
    long["draw"] = (long["goals_for"] == long["goals_against"]).astype(int)
    long["points"] = long["win"] * 3 + long["draw"] * 1
    return long.sort_values("date").reset_index(drop=True)

long = to_long(matches)

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING — pre-match historical strength (NO LEAKAGE)
#    Every feature uses .shift() / expanding windows so a match never
#    "sees" its own result or any future result.
# ---------------------------------------------------------------------------
long = long.sort_values(["team", "date"]).reset_index(drop=True)
g = long.groupby("team")

# Cumulative WC experience BEFORE this match
long["wc_matches_played"] = g.cumcount()
long["wc_wins_prior"] = g["win"].apply(lambda s: s.shift().expanding().sum()).reset_index(level=0, drop=True)
long["wc_winrate_prior"] = g["win"].apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True)
long["wc_gf_avg_prior"] = g["goals_for"].apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True)
long["wc_ga_avg_prior"] = g["goals_against"].apply(lambda s: s.shift().expanding().mean()).reset_index(level=0, drop=True)

# Recent form: points in last 5 WC matches (rolling, shifted)
long["form_last5"] = g["points"].apply(
    lambda s: s.shift().rolling(5, min_periods=1).sum()
).reset_index(level=0, drop=True)

long = long.fillna(0)

# ---------------------------------------------------------------------------
# 4. HEAD-TO-HEAD — prior meetings win rate between the two teams
# ---------------------------------------------------------------------------
long = long.sort_values("date").reset_index(drop=True)
h2h = {}
h2h_winrate = []
for _, r in long.iterrows():
    key = (r["team"], r["opponent"])
    played, won = h2h.get(key, (0, 0))
    h2h_winrate.append(won / played if played > 0 else 0.5)
    h2h[key] = (played + 1, won + r["win"])
long["h2h_winrate_prior"] = h2h_winrate

# ---------------------------------------------------------------------------
# 5. RESHAPE BACK TO MATCH LEVEL — home features vs away features + target
# ---------------------------------------------------------------------------
home_feat = long[long["is_home_listed"] == 1].set_index("match_key")
away_feat = long[long["is_home_listed"] == 0].set_index("match_key")

feat_cols = ["wc_matches_played", "wc_wins_prior", "wc_winrate_prior",
             "wc_gf_avg_prior", "wc_ga_avg_prior", "form_last5", "h2h_winrate_prior"]

model_df = pd.DataFrame(index=home_feat.index)
model_df["year"] = home_feat["year"]
model_df["tournament"] = home_feat["tournament"]
model_df["stage"] = home_feat["stage"]
model_df["home_team"] = home_feat["team"]
model_df["away_team"] = home_feat["opponent"]
for c in feat_cols:
    model_df[f"home_{c}"] = home_feat[c]
    model_df[f"away_{c}"] = away_feat[c]
# Differential features (often the strongest signal)
for c in feat_cols:
    model_df[f"diff_{c}"] = home_feat[c] - away_feat[c]

# TARGET: did the home-listed team win? (multiclass also possible)
model_df["home_win"] = home_feat["win"].astype(int)
model_df["result"] = np.where(home_feat["win"] == 1, "win",
                       np.where(home_feat["draw"] == 1, "draw", "loss"))

model_df = model_df.dropna().reset_index()
model_df.to_csv(OUT / "model_dataset.csv", index=False)
print(f"Model dataset: {model_df.shape[0]} matches, {model_df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 6. WRITE TO SQLITE (dev) — clean relational tables
# ---------------------------------------------------------------------------
con = sqlite3.connect(OUT / "worldcup.db")
matches[["key_id","tournament_name","year","stage_name","match_date",
         "home_team_name","away_team_name","home_team_score","away_team_score",
         "result"]].rename(columns={"key_id":"match_id"}).to_sql(
    "matches", con, if_exists="replace", index=False)
long.to_sql("team_match_long", con, if_exists="replace", index=False)
model_df.to_sql("model_features", con, if_exists="replace", index=False)
con.commit()
con.close()
print("SQLite DB written to output/worldcup.db")
