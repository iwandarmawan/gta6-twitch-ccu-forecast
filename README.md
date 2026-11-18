# GTA VI Twitch CCU forecast — analog-launch forecaster (v2)

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/iwandarmawan/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/iwandarmawan/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An entry for [make.fun gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first hour's print, as
five quantiles an hour.

## Method

Analog launches, conditioned on the size of hour 1. For every comparable
past launch with a complete release day in the case file, the model takes
its log-ratio curve `ln(viewers_h / viewers_1)` for hours 2–24. Each hour's
centre is a kernel-weighted average of those curves, weighted by how close
the comparable's own hour-1 viewers sit to the target's (Gaussian kernel,
bandwidth 1.0 in log units) — a bigger hour 1 means the launch already
unlocked and the curve stays flatter, a small one means the unlock is still
rolling in and the day ramps. The centre is a 10%-trimmed weighted mean,
blended 85/15 with a per-hour least-squares fit of the log ratio on
ln(hour-1 viewers), which captures the size-to-shape slope the kernel mean
alone misses. The spread is the weighted standard deviation around the
centre, floored at 0.05 and scaled by 0.8, both smoothed one hour each way.
Quantiles are `p1 * exp(centre + z_tau * sigma)`.

## Local evaluation

Leave-one-out over the 37 comparables with complete release-day data in
the kit (each launch forecast from its hour-1 print using the other 36):

- this model (v2): mean pinball loss **0.251** on regime-1 targets (hour 1
already unlocked, the GTA VI regime), **0.379** over all 37
- v1, same protocol: **0.260** / **0.387** (reported 0.258 / 0.393)
- house persistence baseline, same protocol: **0.351** / **0.930**

So the analog forecast beats the baseline clearly on the evaluation set,
by about 26% on the launches shaped like GTA VI's. The practice cases'
answers are not in the kit, so no practice score exists locally; the
server grades that.

## Robustness

An invalid case scores as if every quantile equalled hour 1's print and
forfeits any prize, so the program never crashes on a case: anything
unexpected falls back to the persistence shape, and the output is validated
before it is written. Fully deterministic (no randomness, no clock, no
network), standard library only, runs in well under a second per case.

## Run it

    python3 main.py <input-dir> <output-dir>

With no arguments it reads `/input` and writes `/output`, as the grader
runs it.
