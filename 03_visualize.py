"""
World Cup 2026 Predictor — Phase 2b: Visualizations & 2026 Team Strength
========================================================================
Produces portfolio-quality charts and a current strength table for the
2026 qualified/contender nations based on their full historical profile.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = Path(__file__).parent / "output"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# Portfolio styling
mpl.rcParams.update({
    "figure.dpi": 120, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
GREEN, RED, NAVY = "#006847", "#CE1126", "#1d3557"  # Mexico flag accent

long = pd.read_csv(OUT / "model_dataset.csv")

# --- Build all-time team strength profile from the long team view in DB ---
import sqlite3
con = sqlite3.connect(OUT / "worldcup.db")
tml = pd.read_sql("SELECT * FROM team_match_long", con)
con.close()

profile = tml.groupby("team").agg(
    matches=("win", "size"), wins=("win", "sum"), draws=("draw", "sum"),
    gf=("goals_for", "mean"), ga=("goals_against", "mean"),
    last_year=("year", "max"),
).reset_index()
profile["winrate"] = profile["wins"] / profile["matches"]
profile["goal_diff_avg"] = profile["gf"] - profile["ga"]
# Strength score: blend of winrate, experience, goal differential (z-scored)
for c in ["winrate", "matches", "goal_diff_avg"]:
    profile[f"z_{c}"] = (profile[c] - profile[c].mean()) / profile[c].std()
profile["strength_score"] = (0.45*profile["z_winrate"] +
                             0.25*profile["z_matches"] +
                             0.30*profile["z_goal_diff_avg"])
profile = profile.sort_values("strength_score", ascending=False)
profile.to_csv(OUT / "team_strength_profile.csv", index=False)

# === FIG 1: Top 15 all-time strength ===
top = profile.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = [RED if t == "Mexico" else NAVY for t in top["team"]]
ax.barh(top["team"], top["strength_score"], color=colors)
ax.set_title("All-Time World Cup Strength Score — Top 15 Nations",
             fontweight="bold", pad=14)
ax.set_xlabel("Composite strength (win rate · experience · goal differential)")
plt.tight_layout(); plt.savefig(FIG / "01_strength_top15.png", bbox_inches="tight"); plt.close()

# === FIG 2: Mexico historical journey ===
mex = tml[tml["team"] == "Mexico"].groupby("year").agg(
    pts=("points", "sum"), matches=("win", "size")).reset_index()
mex["ppg"] = mex["pts"] / mex["matches"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mex["year"], mex["ppg"], marker="o", color=GREEN, linewidth=2.2)
ax.axhline(mex["ppg"].mean(), ls="--", color=RED, alpha=0.6,
           label=f"Mexico avg = {mex['ppg'].mean():.2f} PPG")
ax.set_title("Mexico at the World Cup — Points Per Game by Tournament",
             fontweight="bold", pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Points per game"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "02_mexico_journey.png", bbox_inches="tight"); plt.close()

# === FIG 3: Feature importance ===
imp = pd.read_csv(OUT / "rf_importance.csv", index_col=0).head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(imp.index, imp["importance"], color=NAVY)
ax.set_title("What Drives a World Cup Result? — Model Feature Importance",
             fontweight="bold", pad=12)
ax.set_xlabel("Random Forest importance")
plt.tight_layout(); plt.savefig(FIG / "03_feature_importance.png", bbox_inches="tight"); plt.close()

print("Figures saved.")
print("\nTOP 10 ALL-TIME STRENGTH:")
print(profile[["team","matches","winrate","goal_diff_avg","strength_score"]].head(10).to_string(index=False))
print("\nMEXICO RANK:", profile.reset_index(drop=True).query("team=='Mexico'").index[0] + 1, "of", len(profile))
