"""
World Cup 2026 Predictor — Phase 2: Modeling & Evaluation
=========================================================
Trains interpretable models to predict match outcomes using a TEMPORAL
train/test split (train <= 2014, test = 2018 & 2022) — the honest way to
evaluate a forecasting model. Compares Logistic Regression vs Random Forest,
reports accuracy / log-loss / Brier score, and inspects feature importance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
import json

OUT = Path(__file__).parent / "output"
df = pd.read_csv(OUT / "model_dataset.csv")

feature_cols = [c for c in df.columns if c.startswith(("home_wc", "away_wc",
                "home_form", "away_form", "home_h2h", "away_h2h", "diff_"))]

# --- Temporal split: honest forecasting evaluation ---
train = df[df["year"] <= 2014]
test = df[df["year"] >= 2018]
X_train, y_train = train[feature_cols], train["home_win"]
X_test, y_test = test[feature_cols], test["home_win"]
print(f"Train: {len(train)} matches (1930-2014) | Test: {len(test)} matches (2018-2022)")
print(f"Baseline (always predict home win): {y_test.mean():.3f}\n")

scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

results = {}

# --- Model 1: Logistic Regression (interpretable) ---
lr = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr_s, y_train)
p_lr = lr.predict_proba(Xte_s)[:, 1]
results["Logistic Regression"] = {
    "accuracy": accuracy_score(y_test, lr.predict(Xte_s)),
    "log_loss": log_loss(y_test, p_lr),
    "brier": brier_score_loss(y_test, p_lr),
    "auc": roc_auc_score(y_test, p_lr),
}

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                            random_state=42).fit(X_train, y_train)
p_rf = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, rf.predict(X_test)),
    "log_loss": log_loss(y_test, p_rf),
    "brier": brier_score_loss(y_test, p_rf),
    "auc": roc_auc_score(y_test, p_rf),
}

print("MODEL COMPARISON (test = WC 2018 + 2022)")
print("-" * 60)
for name, m in results.items():
    print(f"{name:22s} acc={m['accuracy']:.3f}  logloss={m['log_loss']:.3f}  "
          f"brier={m['brier']:.3f}  auc={m['auc']:.3f}")

# --- Feature importance / coefficients ---
print("\nTOP DRIVERS (Logistic Regression coefficients):")
coefs = pd.Series(lr.coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
for f, v in coefs.head(8).items():
    print(f"  {f:28s} {v:+.3f}")

print("\nTOP DRIVERS (Random Forest importance):")
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for f, v in imp.head(8).items():
    print(f"  {f:28s} {v:.3f}")

# Save artifacts for the dashboard
coefs.to_csv(OUT / "lr_coefficients.csv", header=["coefficient"])
imp.to_csv(OUT / "rf_importance.csv", header=["importance"])
with open(OUT / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nArtifacts saved (coefficients, importances, metrics).")
