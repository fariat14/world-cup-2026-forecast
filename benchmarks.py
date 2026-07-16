"""Reproduces the benchmark table in README.md.

Compares the 3-feature logistic regression against feature-upgraded and
XGBoost variants on the same time-based holdout. Spoiler: they tie.

Needs: pip install xgboost (in addition to requirements.txt)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from wc2026_final_predictor import (
    FORM_WINDOW,
    HOME_ADV,
    K_BY_TOURNAMENT,
    K_DEFAULT,
    START_ELO,
    TEST_FROM,
    TRAIN_FROM,
    load_data,
    mov_multiplier,
)


def compute_features_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Same Elo pass as the main script, plus two extra pre-match
    features used by the upgraded variants: rolling Elo change over the
    last 5 matches (opponent-adjusted form) and each side's rating for
    the elo_sum feature."""
    ratings: dict = {}
    recent: dict = {}
    elo_deltas: dict = {}

    pre = {"elo_h": [], "elo_a": [], "form_h": [], "form_a": [], "ec_h": [], "ec_a": []}

    def form(team):
        hist = recent.get(team, [])[-FORM_WINDOW:]
        return float(np.mean([gf - ga for gf, ga in hist])) if hist else 0.0

    def elo_change(team):
        hist = elo_deltas.get(team, [])[-5:]
        return float(np.sum(hist)) if hist else 0.0

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, START_ELO)
        ra = ratings.get(a, START_ELO)

        pre["elo_h"].append(rh)
        pre["elo_a"].append(ra)
        pre["form_h"].append(form(h))
        pre["form_a"].append(form(a))
        pre["ec_h"].append(elo_change(h))
        pre["ec_a"].append(elo_change(a))

        hadv = 0.0 if row.neutral else HOME_ADV
        expected = 1.0 / (1.0 + 10 ** (-((rh + hadv) - ra) / 400.0))
        if row.home_score > row.away_score:
            result = 1.0
        elif row.home_score < row.away_score:
            result = 0.0
        else:
            result = 0.5
        k = K_BY_TOURNAMENT.get(row.tournament, K_DEFAULT)
        k *= mov_multiplier(abs(int(row.home_score - row.away_score)))
        delta = k * (result - expected)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
        elo_deltas.setdefault(h, []).append(delta)
        elo_deltas.setdefault(a, []).append(-delta)
        recent.setdefault(h, []).append((row.home_score, row.away_score))
        recent.setdefault(a, []).append((row.away_score, row.home_score))

    df = df.copy()
    for col, vals in pre.items():
        df[col] = vals
    return df


def main():
    df = load_data()
    df = compute_features_extended(df)

    m = df[df.date >= TRAIN_FROM].copy()
    m["hadv"] = np.where(m.neutral, 0.0, 1.0)
    m["elo_diff"] = m.elo_h - m.elo_a
    m["elo_sum"] = (m.elo_h + m.elo_a) / 1000
    m["form_diff"] = m.form_h - m.form_a
    m["ec_diff"] = m.ec_h - m.ec_a
    m["outcome"] = np.select(
        [m.home_score > m.away_score, m.home_score < m.away_score], [1, 0], default=-1
    )

    dec = m[m.outcome != -1]
    train = dec[dec.date < TEST_FROM].copy()
    test = dec[dec.date >= TEST_FROM]

    # sample weights: time decay (8-year half-life) x match importance
    years_back = (pd.Timestamp(TEST_FROM) - train.date).dt.days / 365.25
    w = (0.5 ** (years_back / 8)) * (train.tournament.map(K_BY_TOURNAMENT).fillna(K_DEFAULT) / K_DEFAULT)

    base = ["elo_diff", "form_diff", "hadv"]
    upgraded = ["elo_diff", "elo_sum", "form_diff", "ec_diff", "hadv"]

    def report(name, clf, feats, weights=None):
        clf.fit(train[feats], train.outcome, sample_weight=weights)
        p = clf.predict_proba(test[feats])[:, 1]
        print(f"{name:40s} brier {brier_score_loss(test.outcome, p):.4f}  "
              f"acc {accuracy_score(test.outcome, p > 0.5):.3f}")

    print(f"holdout: {len(test)} decisive matches, {TEST_FROM}+\n")
    report("logistic, 3 features (main model)", LogisticRegression(max_iter=1000), base)
    report("logistic, upgraded features", LogisticRegression(max_iter=1000), upgraded)
    report("logistic, upgraded + weights", LogisticRegression(max_iter=1000), upgraded, w.values)

    try:
        from xgboost import XGBClassifier
        report(
            "xgboost, upgraded + weights",
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"),
            upgraded, w.values,
        )
    except ImportError:
        print("xgboost not installed; skipping (pip install xgboost)")


if __name__ == "__main__":
    main()
