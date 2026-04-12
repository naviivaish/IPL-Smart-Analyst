"""
Train match-winner classifier and save artifacts for the Streamlit app.

Run from project root:
    python model/train_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Project root (parent of model/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils  # noqa: E402


def build_feature_matrix(m: pd.DataFrame, team_enc, toss_enc, venue_enc) -> pd.DataFrame:
    """Apply fitted encoders and return numeric X."""
    out = m.copy()
    out["team1_enc"] = team_enc.transform(out["team1"].astype(str))
    out["team2_enc"] = team_enc.transform(out["team2"].astype(str))
    out["toss_winner_enc"] = team_enc.transform(out["toss_winner"].astype(str))
    out["toss_decision_enc"] = toss_enc.transform(out["toss_decision"].astype(str))
    out["venue_enc"] = venue_enc.transform(out["venue"].astype(str))
    feature_cols = [
        "team1_enc",
        "team2_enc",
        "toss_winner_enc",
        "toss_decision_enc",
        "venue_enc",
        "team_win_pct_team1",
        "team_win_pct_team2",
        "team_bat_sr_1",
        "team_bat_sr_2",
        "team_bowl_econ_1",
        "team_bowl_econ_2",
        "toss_advantage",
    ]
    return out[feature_cols]


def main() -> None:
    print("Loading data...")
    matches = utils.load_matches()
    deliveries = utils.load_deliveries()
    m = utils.prepare_model_table(matches, deliveries)

    # Encoders fitted on all labels so the app never sees an "unknown" class at train time
    team_enc = LabelEncoder()
    all_teams = pd.concat([m["team1"], m["team2"], m["toss_winner"], m["winner"]]).astype(str)
    team_enc.fit(all_teams)

    toss_enc = LabelEncoder()
    toss_enc.fit(m["toss_decision"].astype(str))

    venue_enc = LabelEncoder()
    m["venue"] = m["venue"].fillna("Unknown").astype(str)
    venue_enc.fit(m["venue"])

    X = build_feature_matrix(m, team_enc, toss_enc, venue_enc)
    y = m["team1_win"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest: strong baseline on mixed categorical/numeric features
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix [rows=true team1_loss, team1_win]:")
    print(np.array(cm))

    win_pct = utils.compute_team_win_percentage(matches)
    bat_sr = utils.team_batting_strike_rate(deliveries)
    bowl_econ = utils.team_bowling_economy(deliveries)

    bundle = {
        "model": clf,
        "team_encoder": team_enc,
        "toss_encoder": toss_enc,
        "venue_encoder": venue_enc,
        "team_win_pct": win_pct,
        "team_bat_sr": bat_sr,
        "team_bowl_econ": bowl_econ,
        "feature_columns": list(X.columns),
        "metrics": {
            "accuracy": float(acc),
            "confusion_matrix": cm,
            "model_name": "RandomForestClassifier",
        },
    }

    out_path = ROOT / "model" / "match_winner_bundle.joblib"
    joblib.dump(bundle, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
