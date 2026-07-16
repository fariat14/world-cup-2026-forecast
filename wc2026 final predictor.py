"""World Cup 2026 predictor.

Computes Elo ratings from full match history (1872-present), trains a
logistic regression on top, and simulates the remaining bracket.
Predictions locked July 14, 2026, before the semifinals concluded.

See README.md for methodology and reasoning behind design choices.

Data: github.com/martj42/international_results (CC0)
Author: Faria Tabassum
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
LOCAL_CACHE = "results.csv"

# Elo parameters. K controls how much one result moves a rating.
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "Copa América": 50,
    "Friendly": 20,
}
K_DEFAULT = 30
HOME_ADV = 80.0       # rating points added to home side's expectation
START_ELO = 1500.0
FORM_WINDOW = 10      # matches used for rolling form
TRAIN_FROM = "1995-01-01"
TEST_FROM = "2022-01-01"

FEATURES = ["elo_diff", "form_diff", "hadv"]


def load_data(path: str = LOCAL_CACHE) -> pd.DataFrame:
    """Load match results, downloading once and caching locally."""
    if not os.path.exists(path):
        print(f"downloading dataset to {path} ...")
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)


def mov_multiplier(margin: int) -> float:
    """Margin-of-victory weight. Diminishing returns above 2 goals so
    blowouts against weak sides don't distort ratings."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def compute_elo_features(df: pd.DataFrame):
    """One pass through history. Features are recorded BEFORE each match
    updates the ratings -- this is the leakage guard. Don't reorder."""
    ratings: dict = {}
    recent: dict = {}  # team -> [(gf, ga), ...]

    pre = {"elo_h": [], "elo_a": [], "form_h": [], "form_a": []}

    def form(team):
        hist = recent.get(team, [])[-FORM_WINDOW:]
        return float(np.mean([gf - ga for gf, ga in hist])) if hist else 0.0

    for row in df.itertuples(index=False):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, START_ELO)
        ra = ratings.get(a, START_ELO)

        pre["elo_h"].append(rh)
        pre["elo_a"].append(ra)
        pre["form_h"].append(form(h))
        pre["form_a"].append(form(a))

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
        recent.setdefault(h, []).append((row.home_score, row.away_score))
        recent.setdefault(a, []).append((row.away_score, row.home_score))

    df = df.copy()
    for col, vals in pre.items():
        df[col] = vals
    return df, ratings, recent


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df.date >= TRAIN_FROM].copy()
    m["hadv"] = np.where(m.neutral, 0.0, 1.0)
    m["elo_diff"] = m.elo_h - m.elo_a
    m["form_diff"] = m.form_h - m.form_a
    m["outcome"] = np.select(
        [m.home_score > m.away_score, m.home_score < m.away_score], [1, 0], default=-1
    )
    return m


def train_and_validate(m: pd.DataFrame) -> LogisticRegression:
    """Time-based split. A random split would leak future info into
    training and flatter every metric."""
    decisive = m[m.outcome != -1]
    train = decisive[decisive.date < TEST_FROM]
    test = decisive[decisive.date >= TEST_FROM]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train[FEATURES], train.outcome)

    p = clf.predict_proba(test[FEATURES])[:, 1]
    print(f"validation on {len(test)} held-out matches ({TEST_FROM}+)")
    print(f"  brier:    {brier_score_loss(test.outcome, p):.4f}  (coin flip = 0.25)")
    print(f"  accuracy: {accuracy_score(test.outcome, p > 0.5):.3f}")
    print(f"  coefs:    {dict(zip(FEATURES, clf.coef_[0].round(4)))}")
    return clf


def empirical_draw_rate(m: pd.DataFrame) -> float:
    """P(level after 90') for close matches at neutral venues."""
    close = m[(m.elo_diff.abs() < 100) & m.neutral & (m.date >= "2000-01-01")]
    return float((close.outcome == -1).mean())


def win_prob(clf, ratings, recent, p_draw, team_a, team_b) -> float:
    """P(team_a wins a knockout match vs team_b at a neutral venue).

    Averages both orderings: the trained intercept absorbs residual home
    bias, so a single call is order-dependent by a few points even with
    hadv=0. Caught this when Argentina-England changed with argument
    order. Symmetrizing removes it.

    Draws after 90' go to ET/pens, resolved near 50/50 with a small tilt
    to the higher-rated side. Shootouts are close to coin flips; not
    pretending otherwise."""
    ra, rb = ratings[team_a], ratings[team_b]
    fa = np.mean([gf - ga for gf, ga in recent[team_a][-FORM_WINDOW:]])
    fb = np.mean([gf - ga for gf, ga in recent[team_b][-FORM_WINDOW:]])
    x_ab = pd.DataFrame([[ra - rb, fa - fb, 0.0]], columns=FEATURES)
    x_ba = pd.DataFrame([[rb - ra, fb - fa, 0.0]], columns=FEATURES)
    p_decisive = (clf.predict_proba(x_ab)[0, 1] + 1 - clf.predict_proba(x_ba)[0, 1]) / 2
    p_pens = 0.5 + np.clip((ra - rb) / 2000, -0.05, 0.05)
    return (1 - p_draw) * p_decisive + p_draw * p_pens


def simulate_bracket(clf, ratings, recent, p_draw, n_sims, seed):
    """Monte Carlo over the remaining bracket as of July 14:
    SF1 France-Spain, SF2 England-Argentina, then the final."""
    rng = np.random.default_rng(seed)
    p_sf1 = win_prob(clf, ratings, recent, p_draw, "France", "Spain")
    p_sf2 = win_prob(clf, ratings, recent, p_draw, "England", "Argentina")
    print(f"\nsemifinals: Spain over France {1 - p_sf1:.1%}, "
          f"Argentina over England {1 - p_sf2:.1%}")

    sf1 = np.where(rng.random(n_sims) < p_sf1, "France", "Spain")
    sf2 = np.where(rng.random(n_sims) < p_sf2, "England", "Argentina")

    champs = np.empty(n_sims, dtype=object)
    for a in ("France", "Spain"):
        for b in ("England", "Argentina"):
            mask = (sf1 == a) & (sf2 == b)
            if not mask.any():
                continue
            p_ab = win_prob(clf, ratings, recent, p_draw, a, b)
            champs[mask] = np.where(rng.random(mask.sum()) < p_ab, a, b)

    print(f"\nchampionship probabilities ({n_sims:,} sims):")
    teams, counts = np.unique(champs, return_counts=True)
    for t, c in sorted(zip(teams, counts), key=lambda x: -x[1]):
        print(f"  {t:<10} {c / n_sims:.1%}")


def post_sf1_title_odds(clf, ratings, recent, p_draw):
    """Title odds updated for Spain beating France in SF1 (July 14).
    Conditions the bracket on the known result rather than resimulating."""
    p_arg = win_prob(clf, ratings, recent, p_draw, "Argentina", "England")
    p_esp_arg = win_prob(clf, ratings, recent, p_draw, "Spain", "Argentina")
    p_esp_eng = win_prob(clf, ratings, recent, p_draw, "Spain", "England")
    spain = p_arg * p_esp_arg + (1 - p_arg) * p_esp_eng
    print("\ntitle odds updated for SF1 result (Spain 2-0 France):")
    print(f"  Spain      {spain:.1%}")
    print(f"  Argentina  {p_arg * (1 - p_esp_arg):.1%}")
    print(f"  England    {(1 - p_arg) * (1 - p_esp_eng):.1%}")
    print(f"  France     0.0%  (plays third-place match)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=LOCAL_CACHE, help="path to results csv")
    parser.add_argument("--sims", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"{len(df):,} matches, {df.date.min().date()} to {df.date.max().date()}")

    df, ratings, recent = compute_elo_features(df)
    m = build_model_frame(df)
    clf = train_and_validate(m)

    p_draw = empirical_draw_rate(m)
    print(f"  p(draw after 90'), close neutral matches: {p_draw:.1%}")

    print("\npossible finals (neutral venue):")
    for a, b in [("France", "England"), ("France", "Argentina"),
                 ("Spain", "England"), ("Spain", "Argentina")]:
        p = win_prob(clf, ratings, recent, p_draw, a, b)
        print(f"  {a} vs {b}: {p:.1%} / {1 - p:.1%}")

    simulate_bracket(clf, ratings, recent, p_draw, args.sims, args.seed)
    post_sf1_title_odds(clf, ratings, recent, p_draw)

    # TODO: backtest on 2018 and 2022 tournaments with calibration curves


if __name__ == "__main__":
    main()
