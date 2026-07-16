# Can a Simple Model Predict the World Cup?

*A football prediction project by someone who doesn't watch football.*

## Why I Did This

As I write this, on July 14, 2026, the World Cup is everywhere. Everyone around me has opinions. I have none, because my football knowledge starts and ends at goals, red cards, and yellow cards. No favorite team either.

Which, it turns out, is useful. I built this model and locked every prediction on July 14, before the semifinals concluded. I want to know if my actual skills (data analysis, statistics, an economics background) can produce a real forecast for a sport I barely follow. If it works, it works because of the method. Not because I know football.

So here's the question: **using nothing but historical match results, how well can you predict who wins a World Cup match? And where does that approach stop working?**

*This is a personal learning project, not betting advice. I'm not a sports analyst. © 2026 Faria Tabassum. All rights reserved.*

## What I Built, Short Version

**The prediction, as of July 14 evening, with Spain through to the final:**

| | Chance of winning the World Cup |
|---|---|
| **Spain** | **53.3%** |
| Argentina | 32.3% |
| England | 14.4% |
| France | 0% (out; plays for third place) |

Spain is the favorite but far from a lock. The most likely final is Spain vs Argentina, and in that specific matchup my model actually has Argentina by a nose. How both of those can be true at once is explained in [How to Read These Numbers](#how-to-read-these-numbers).

**How it was built:** a strength rating (Elo) for every national team, computed by replaying 49,505 international matches from 1872 to now, with a logistic regression on top that turns rating and form gaps into win probabilities.

**Does it work?**

- **78.3% accuracy** on 3,597 matches the model had never seen
- Original forecast, locked before the semifinals: Argentina ~33%, Spain ~28%, France ~25%, England ~15%
- First live test: my model rated the France-Spain semifinal 52/48 for Spain, barely more than a coin flip. A lazier model I built for comparison, using only this tournament's stats, confidently picked France at 58%. Spain won 2-0. The confident model was the wrong one.
- I built three features that failed, and honestly they taught me more than the ones that worked

## Table of Contents

1. [The Data](#the-data)
2. [The Variables](#the-variables)
3. [The Model, and Why This One](#the-model-and-why-this-one)
4. [What Others Have Done](#what-others-have-done)
5. [How It Works](#how-it-works)
6. [Results](#results)
7. [How to Read These Numbers](#how-to-read-these-numbers)
8. [Three Features I Deleted](#three-features-i-deleted)
9. [What This Model Can't See](#what-this-model-cant-see)
10. [What I'd Build Next](#what-id-build-next)
11. [Run It Yourself](#run-it-yourself)
12. [What I'm Taking Away](#what-im-taking-away)

## The Data

One source: [martj42/international_results](https://github.com/martj42/international_results). It's a free, actively maintained record of every international match since 1872. Each row is one match: date, teams, score, tournament, city, and whether the venue was neutral.

I picked it because it's the standard for this problem, it's updated within days of matches being played, and it costs nothing. No scraping, no paid API, no merging multiple sources. Everything else in this project is calculated from this one file. One source means one point of failure, and anyone can reproduce my numbers exactly.

| | |
|---|---|
| Total matches | 49,505 |
| Date range | Nov 1872 to present |
| Teams | 336 |
| Used for training (1995 onward) | 29,393 matches |
| Outcomes since 1995 | 48.5% home win, 23.3% draw, 28.2% away win |
| Average goals per match | 2.77 |
| Neutral venues | 28.4% |

Look at that outcome split for a second. Home teams win nearly half of all matches. That's your first finding before any modeling happens: a football model that ignores venue is broken on arrival.

## The Variables

Three. Each one uses only information that existed before kickoff, which matters more than it sounds (explained below).

**1. `elo_diff`, the strength gap.** Every team carries a strength score. Think credit score, but for football teams: beat a strong side and your score jumps, lose to a weak one and it drops hard. I computed these by replaying all 49,505 matches in order. The variable is just Team A's score minus Team B's. This one does almost all the work.

**2. `form_diff`, the recent-form gap.** Average goal difference over each team's last 10 matches. Catches "who's hot right now," which the slower-moving Elo score partly misses.

**3. `hadv`, home advantage.** 1 at home, 0 on neutral ground. Every 2026 knockout match is on neutral ground in North America, so this is always 0 for my predictions. The model still needs it to learn correctly from 150 years of history where venue mattered a lot.

The model outputs one number: the probability Team A beats Team B, assuming someone wins. Draws get handled separately.

## The Model, and Why This One

Logistic regression. Probably the least impressive model I could have named, and I chose it on purpose.

Here's the logic. Published work keeps finding that the Elo gap does nearly all the predictive work in football (one study measured it at roughly 100x the influence of the next feature). When one variable dominates like that, a linear model captures the whole signal and a fancy one has nothing extra to find.

I didn't just take that on faith though. I benchmarked XGBoost, the model that usually wins Kaggle competitions, with richer features and sample weighting, on the same test set. It beat my logistic regression by 0.0007 Brier. That's noise.

And there's a practical reason: I can read every coefficient. One of my rejected features got caught precisely because its coefficient came out with an impossible sign. A black-box model would have swallowed it silently.

## What Others Have Done

I went through a fair amount of prior work before settling on my approach:

- **[DataCamp's 2026 World Cup project](https://www.datacamp.com/tutorial/fifa-world-cup-2026-winner-prediction)** (June 2026). The closest comparison to mine. The author benchmarked ten models across five families on a 347-match holdout. XGBoost won by a hair, the top five were basically tied, deep learning came in last, and Elo difference dominated everything else. My own benchmark reproduced this pattern independently, which was reassuring.
- **Dixon & Coles (1997), Karlis & Ntzoufras (2004).** The classical statistics route: model each team's goal count directly with Poisson distributions. This is what the "supercomputer predictions" in the press are usually built on. My tournament simulation uses a simplified version.
- **[Elo++](https://arxiv.org/abs/1012.4571)**, the system that won Kaggle's "Elo vs. the Rest of the World" chess competition. The winner didn't beat Elo with a fancier model. He beat it with regularization, while the leaderboard filled up with complex models that overfit. That warning shaped my "keep it simple" decision.
- **Player-level approaches** like [this 2025 paper](https://arxiv.org/abs/2505.01902), which blend individual player stats into team predictions. More accurate in principle. Also requires data I don't have, so it's on my future list instead.
- **Prediction markets.** Kaggle's 2026 competition literally frames its benchmark as "beat Dixon-Coles and the prediction markets." Betting odds absorb injuries, lineups, and information no public dataset has. That's the honest ceiling for a project like this one.

## How It Works

**Compute Elo.** Everyone starts at 1500 in 1872. Replay history match by match, updating after each result. Plain Elo (the chess version) only knows who won, but football results say more than that, so I extended it three ways: World Cup matches move ratings 3x more than friendlies, winning 4-0 counts more than 1-0 (with diminishing returns so blowouts against minnows don't distort), and home teams get +80 added to their expected performance, zeroed at neutral venues.

**Block data leakage.** For every historical match, I record the features *before* that match's result updates the ratings. This sounds like a technicality. It's actually the most common way prediction projects quietly cheat: a model that knows a team's post-match rating while "predicting" that match looks brilliant in testing and useless in real life.

**Train.** Logistic regression, 27,000+ decisive matches from 1995 onward.

**Validate the honest way.** Train on everything before 2022, test on the 3,597 matches after. Not a random split. Random splits let the model peek at the future, and your metrics come out flattering and wrong.

**Handle draws.** Knockout matches can't end level, but 90 minutes can, and about 28% of close neutral-venue matches do. In my simulation those go to extra time and penalties, settled near 50/50 with a small tilt to the stronger team. Penalty shootouts really are close to coin flips. I'm not going to pretend my model knows who converts from the spot.

**Simulate the tournament.** Monte Carlo: play the remaining bracket 500,000 times, sampling each match result from the model's win probability for that pairing. The share of runs a team wins is its championship probability.

## Results

| Model | Brier score (lower = better) | Accuracy |
|---|---|---|
| **Logistic regression, 3 features (mine)** | **0.1499** | **78.3%** |
| Plus Elo sum and opponent-adjusted form | 0.1501 | 78.4% |
| Plus time-decay and importance weights | 0.1498 | 78.3% |
| XGBoost with all upgrades | 0.1492 | 78.6% |

Brier score measures probability quality. 0.25 is a coin flip.

Four very different setups landing within 0.001 of each other is the same thing the ten-model DataCamp study found. The ceiling comes from the data, not the algorithm. Once the Elo gap is in, everything else is garnish.

**Tournament forecast, locked before the semifinals:** Argentina 33.0%, Spain 27.7%, France 24.7%, England 14.6%. Run the script and these are the numbers it prints, including the updated post-semifinal table at the top of this README. (Computed with data through July 11, 2026. The dataset updates as matches are played, so ratings and probabilities will drift slightly on later data. That's expected, and it's why every forecast here carries a date.)

**Scorecard (updated July 14, evening):** semifinal one, my model rated Spain over France at just 52/48, barely more than a coin flip. It refused to commit, and I think that was the honest answer. The quick tournament-stats-only model I built for comparison had no such doubts: France at 57.8%, on the strength of their goal difference this tournament. Spain won 2-0. One result proves nothing about either model, but it's a clean illustration of the difference between a calibrated model and a confident one. The confident one was leaning on schedule-inflated stats, and the section after next explains why.

## How to Read These Numbers

A note on bias before anything else. The model's learned parameters have never seen this tournament: the regression was trained only on matches played before 2022. This World Cup's results do feed the inputs (they update team ratings and recent form, exactly the way every other match in 154 years does), but no feature treats this tournament specially, nothing was manually adjusted for any team, and no opinion of mine enters the pipeline anywhere. I don't have opinions about football teams to insert.

The single biggest way to misread this project is to turn probabilities into picks. "Argentina 33%" does not mean "Argentina will win." It means that if this tournament were played many times, Argentina lifts the trophy in about one of every three. In the other two, the forecast wasn't wrong; the less likely thing happened, which is most of what happens in football.

Probabilities also move as matches resolve. Spain winning semifinal one changes everything downstream. Conditioning on that result:

| | Title chance (updated July 14, evening) |
|---|---|
| Spain | **53.3%** |
| Argentina | 32.3% |
| England | 14.4% |
| France | 0% (plays the third-place match) |

And here's my favorite wrinkle, because two true statements sound like they contradict each other:

- The **most likely champion** is Spain, at 53.3%.
- The **most likely complete scenario** (about a 19% chance) is: Argentina beats England in the semifinal, Argentina beats Spain in the final, France takes third, England fourth.

Both are correct. In the single most probable final matchup, Spain vs Argentina, my model has Argentina by a nose, 51.5 to 48.5. But Spain's overall title chance adds up across *both* possible opponents, and against England, Spain is a heavy 61.4% favorite. So Spain is the best bet for the trophy even while being a slight underdog in the most likely final. If that feels like a paradox, sit with it for a minute. Aggregated probability and single-path probability answer different questions, and mixing them up is one of the most common mistakes people make reading forecasts, in sports and everywhere else.

## Three Features I Deleted

**Rolling goals scored.** Its coefficient came out negative, which would mean scoring more goals makes you worse. Obviously wrong, so I dug in: it overlaps heavily with rolling goal difference, which was already in the model (multicollinearity). Deleted it. Brier score moved by 0.0001. When a coefficient's sign makes no sense, check whether your features overlap. It's the fastest tell there is.

**Current-tournament performance.** The intuition felt right: surely a team's results in *this* World Cup should count. The test said no. Overall metrics didn't move, and on major-tournament matches specifically (the exact situation the feature was built for) it made things *worse*, Brier 0.159 to 0.162. Two reasons. The information was already in the model twice: tournament matches update Elo at the highest weight, and by the knockout rounds the form window is mostly tournament games anyway. And raw tournament stats hide schedule strength. At the semifinal stage, Argentina's +11 goal difference came against a soft draw. Spain's +10 came through Portugal, Belgium, and Uruguay. Raw numbers call those equal. They're not, and Elo knows it.

**Opponent-adjusted tournament form,** the "fixed" version of the one above. Rolling Elo change over the last 5 matches, so form that accounts for who you played. Coefficient: 0.0001. Nothing there, and the reason is almost funny: Elo *is* accumulated opponent-adjusted form. I was trying to hand the model information it already had.

**And one bug worth confessing.** While updating predictions after semifinal one, I noticed Argentina vs England gave a different probability than England vs Argentina, off by about four points. Order shouldn't matter at a neutral venue. The cause: the regression's intercept soaks up residual home bias from 150 years of training data, so it leaks into predictions even with the home flag set to zero. The fix was to score every matchup both ways and average. Small bug, easy fix, but it only surfaced because I checked a number I had no reason to doubt. Worth doing.

## What This Model Can't See

- **Players.** Injuries, suspensions, one teenager having the tournament of his life. Invisible to a results-only model. It's why player-based models like Opta's rated France higher than mine did.
- **Parameter sensitivity.** My Elo weights rank Argentina a hair above Spain. eloratings.net's weights rank Spain first. Perfectly reasonable parameter choices flip the top spot, so don't over-read any single rating system, including mine.
- **Penalty shootouts** are near coin flips here, on purpose.
- **The 78% wall.** Football is low-scoring and luck-heavy. One deflection flips a match. Nobody's results-only model does much better than this, and betting markets stay the practical ceiling because they see what models can't.

## What I'd Build Next

- Goals-based Poisson regression (the Dixon-Coles route) instead of win/loss, which unlocks proper scoreline predictions and group-stage simulation
- Backtests on 2018 and 2022 with calibration curves
- Player-level features from something like API-Football
- A pipeline that retrains itself as results come in

## Run It Yourself

```bash
pip install pandas numpy scikit-learn
python wc2026_final_predictor.py
```

Downloads the latest data, rebuilds all ratings from 1872, trains, validates, prints predictions. Under two minutes.

To reproduce the benchmark table (needs `pip install xgboost`):

```bash
python benchmarks.py
```

## What I'm Taking Away

A three-variable model built by someone who can't name a single formation called match winners 78% of the time. And in its first live test, it did something I've come to respect more than a correct pick: it looked at France versus Spain, saw a coin flip, and said so, while my quick-and-dirty comparison model confidently backed France on stats that turned out to be schedule-inflated. Spain won. The method did that, not football knowledge: one clean dataset, leakage-safe features, honest validation, and deleting everything that didn't earn its spot.

Most of the work in this project was figuring out what to leave out. I suspect that's true of more than just football models.

---
© 2026 Faria Tabassum. All rights reserved. Not betting advice.
