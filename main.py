"""Forecast GTA VI's launch-day Twitch viewers, hour by hour, from analog launches.

The entry is a program, not a model call: it reads one case file
(makefun/twitch-ccu-case/v1) from the input directory and writes one
forecast file per case to the output directory. Each forecast gives five
quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of the category's concurrent
viewers for hours 2 to 24 of the release day, scaled from hour 1's print.

Method — analog curves, conditioned on the size of hour 1
----------------------------------------------------------
Every comparable in the case is a past big launch with hourly Twitch counts
from 14 days before its release day to 3 days after it. For each comparable
with a complete release day, take its log-ratio curve

    lr_h = ln(viewers_h / viewers_1),  h = 2..24.

Launch-day curves split into two shapes: launches already unlocked at hour 1
(a substantial hour-1 print) move modestly afterwards, while launches whose
unlock falls later in the UTC day explode from a tiny hour 1. The size of
hour 1 therefore predicts the shape: a bigger hour 1 means a flatter curve.
So each hour's centre is a kernel-weighted average of the comparables'
log-ratios, with weights falling off in ln(hour-1 viewers) around the
target's own hour-1 print (Gaussian kernel, bandwidth 1.0 in log units).
The centre is a 10%-trimmed weighted mean (robust to the odd launch whose
curve is an outlier for its size), and the spread is the weighted standard
deviation of the log-ratios around that centre — wide enough to stay honest
about how much launch curves differ. Both are smoothed one hour each way,
because real curves move smoothly through the day.

    q_tau(h) = p1 * exp(centre_h + z_tau * sigma_h)

with p1 the hour-1 print and z the standard normal's quantiles. The final
case is Grand Theft Auto VI, which unlocks at local midnight rolling around
the world (New Zealand first, the UK at 00:00 UTC, the US at 05:00 and
08:00 UTC), so hour 1 already holds the UK and Australasian audience and
the day ramps as the US unlocks — the shape the analog pool's
rolling-midnight launches trace.

Robustness comes first: an invalid case scores as if every quantile equalled
hour 1's print and forfeits any prize, so the program never crashes on a
case. Anything unexpected (missing data, a malformed comparable, no usable
comparable at all) falls back to the house persistence shape — every hour
at hour 1's print with a widening spread — and the output is validated
before it is written. The run is fully deterministic: no randomness, no
clock, no network, standard library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Kernel bandwidth on ln(hour-1 viewers): how far, in log units, a
# comparable's launch size may sit from the target's before its weight
# fades. 1.0 keeps roughly the launches within a factor e either way.
BANDWIDTH = 1.0
# Trimmed weighted mean: drop this share of the weight from each tail of
# the log-ratio distribution before averaging.
TRIM = 0.10
# The spread never goes below this, in log units.
SIGMA_FLOOR = 0.05
# Fallback persistence spread, in log units per square-root hour.
FALLBACK_SIGMA = 0.15
# Largest viewer count the output format allows.
MAX_VIEWERS = 1_000_000_000


def parse_time(text):
    """An ISO 8601 UTC timestamp into a datetime (handles the trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(value):
    """value rounded to 2 decimals, half up, as a plain number.

    Whole numbers come back as ints, so the JSON writes 1240, not 1240.0.
    """
    scaled = math.floor(value * 100 + 0.5) / 100
    return int(scaled) if float(scaled).is_integer() else scaled


def smooth_hourly(values):
    """A 3-hour moving average over the per-hour values (hours 2..24).

    The first and last hours average over the neighbours they have.
    """
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - 1)
        hi = min(len(values), i + 2)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def release_day(comparable):
    """The comparable's 24 release-day viewer counts, or None.

    Hours are counted from the comparable's day_start (00:00 UTC of its
    release date). A comparable counts only when every hour 1..24 has a
    point with a positive count — the Grand Theft Auto VI category's own
    pre-release history, for instance, has no release day yet and is left
    out, as is any launch SullyGnome sampled sparsely.
    """
    try:
        day = parse_time(comparable["day_start"])
        by_hour = {}
        for point in comparable["hourly"]:
            hour = int((parse_time(point["from"]) - day).total_seconds() // 3600) + 1
            if 1 <= hour <= 24 and point["viewers"] is not None:
                by_hour[hour] = max(1, point["viewers"])
        if len(by_hour) < 24:
            return None
        return [by_hour[h] for h in range(1, 25)]
    except (KeyError, TypeError, ValueError):
        return None


def analog_pool(case):
    """Usable comparables: (hour-1 viewers, log-ratio curve hours 2..24)."""
    pool = []
    for comparable in case.get("comparables") or []:
        series = release_day(comparable)
        if series is None or series[0] <= 0:
            continue
        first = series[0]
        pool.append((first, [math.log(v / first) for v in series[1:]]))
    return pool


def trimmed_weighted_mean(points, trim):
    """Weighted mean of the values after dropping `trim` of the weight
    from each tail. `points` is a list of (value, weight), unsorted."""
    ordered = sorted(points)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    low_cut = trim * total
    high_cut = (1.0 - trim) * total
    acc = 0.0
    weighted_sum = 0.0
    used_weight = 0.0
    for value, weight in ordered:
        acc += weight
        if acc <= low_cut:
            continue
        if acc - weight >= high_cut:
            break
        weighted_sum += value * weight
        used_weight += weight
    if used_weight <= 0:
        return ordered[len(ordered) // 2][0]
    return weighted_sum / used_weight


def weighted_spread(points, centre):
    """Weighted standard deviation of the values around `centre`."""
    total = sum(w for _, w in points)
    if total <= 0:
        return SIGMA_FLOOR
    var = sum(w * (v - centre) ** 2 for v, w in points) / total
    return max(math.sqrt(var), SIGMA_FLOOR)


def analog_curves(pool, p1):
    """Per-hour (centre, sigma) of the log-ratio to hour 1's print.

    Each comparable is weighted by how close its launch size (its own
    hour-1 viewers) sits to the target's, in log units.
    """
    log_p1 = math.log(p1)
    centres = []
    sigmas = []
    for h in range(23):
        points = []
        for first, ratios in pool:
            weight = math.exp(-0.5 * ((math.log(first) - log_p1) / BANDWIDTH) ** 2)
            points.append((ratios[h], weight))
        centre = trimmed_weighted_mean(points, TRIM)
        centres.append(centre)
        sigmas.append(weighted_spread(points, centre))
    return smooth_hourly(centres), smooth_hourly(sigmas)


def hour_quantiles(p1, centre, sigma):
    """The five quantiles for one hour, rounded and clamped to the format."""
    row = []
    for z in Z:
        q = p1 * math.exp(centre + z * sigma)
        q = min(max(q, 0.01), MAX_VIEWERS)
        row.append(round2(q))
    # Keep the row non-decreasing after rounding.
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def persistence_rows(p1, hours):
    """Fallback forecast: every hour at hour 1's print, spread widening
    like a random walk in log space. Used when the analog pool is empty
    or anything about a case fails — a valid forecast beats no forecast."""
    rows = []
    for hour in hours:
        sigma = FALLBACK_SIGMA * math.sqrt(max(hour - 1, 0))
        rows.append({"hour": hour, "q": hour_quantiles(p1, 0.0, sigma)})
    return {"hours": rows}


def hour_one_print(case):
    """Hour 1's viewer count, falling back to the latest history point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1, row["viewers"])
    for key in ("history", "history_hourly"):
        points = case.get(key) or []
        for point in reversed(points):
            if point.get("viewers") is not None:
                return max(1, point["viewers"])
    return 1


def forecast_case(case):
    """One case's forecast. Never raises: on any failure it returns the
    persistence fallback for that case's hours."""
    try:
        hours = [int(row["hour"]) for row in case["forecast"]]
        if sorted(hours) != list(range(2, 25)):
            raise ValueError("forecast hours are not 2..24")
        p1 = hour_one_print(case)
        pool = analog_pool(case)
        if not pool:
            raise ValueError("no usable analog pool")
        centres, sigmas = analog_curves(pool, p1)
        rows = []
        for i, hour in enumerate(hours):
            rows.append({"hour": hour, "q": hour_quantiles(p1, centres[i], sigmas[i])})
        return {"hours": rows}
    except Exception:
        try:
            hours = [int(row["hour"]) for row in case.get("forecast") or []]
            hours = [h for h in hours if 2 <= h <= 24]
            p1 = hour_one_print(case)
        except Exception:
            hours, p1 = list(range(2, 25)), 1
        if not hours:
            hours = list(range(2, 25))
        return persistence_rows(p1, hours)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            # An unreadable case still gets a forecast file: the grader
            # scores a missing file far worse than a persistence one.
            forecast = persistence_rows(1, list(range(2, 25)))
            (OUTPUT_DIR / f"{path.stem}.out").write_text(
                json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            continue
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
