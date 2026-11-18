"""GTA VI launch-day Twitch viewers, hour by hour (hours 2-24).

Analog-launch forecaster, v2. For every past launch with a complete
release day in the case file, take its log-ratio curve lr_h = ln(v_h / v_1)
(hours 2..24, relative to its own hour-1 print). To forecast a new launch
whose hour-1 print is p1:

  * per hour, a Gaussian kernel on ln(hour-1 viewers) around ln(p1)
    (bandwidth 1.0) weights the comparables by how similar in size they are;
  * the centre is a 10%-trimmed kernel-weighted mean of the log ratios,
    blended 85/15 with a per-hour ordinary-least-squares fit of the log
    ratio on ln(hour-1 viewers) -- the size-to-shape slope is real (bigger
    hour 1 means a flatter day), and the small OLS share captures it without
    letting any single explosive launch dominate;
  * the spread is the kernel-weighted standard deviation of the log ratios
    around the centre, floored at 0.05 and scaled by 0.8 (the raw spread
    over-covers: the 90% band covered ~98% of hours on the practice board);
  * centre and spread are smoothed over neighbouring hours (+/-1h);
  * quantiles are q_tau = p1 * exp(centre + z_tau * sigma) with the
    standard normal quantiles.

If anything about a case is unusable (no comparables, bad numbers, unreadable
input), the case falls back to the house persistence shape, so the output
is always a valid forecast and the program never crashes. Standard library
only, no network, fully deterministic (same input -> byte-identical output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
BANDWIDTH = 1.0
TRIM = 0.10
OLS_BLEND = 0.15
SIGMA_SCALE = 0.8
SIGMA_FLOOR = 0.05
MAX_VIEWERS = 1_000_000_000


def _parse_time(text):
    """Parse 'YYYY-MM-DDTHH:MM:SSZ' to seconds since the epoch (UTC)."""
    try:
        moment = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None
    return moment.replace(tzinfo=timezone.utc).timestamp()


def _number(value):
    """Return value as a finite float, or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (math.inf, -math.inf):
        return None
    return out


def load_case(path):
    """Read one case file. Returns (case_id, p1, pool) where pool is a list
    of (hour1_viewers, [log-ratio hour 2 .. hour 24]) for every comparable
    with a complete 24-hour release day. GTA VI's own pre-release history
    has no release day yet, so it drops out here on its own."""
    with open(path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    case_id = os.path.splitext(os.path.basename(path))[0]
    p1 = None
    known = case.get("known") or []
    if known:
        p1 = _number(known[0].get("viewers"))
    if p1 is None or p1 <= 0:
        p1 = 1.0
    pool = []
    for comp in case.get("comparables") or []:
        start = _parse_time(comp.get("day_start"))
        if start is None:
            continue
        by_hour = {}
        for point in comp.get("hourly") or []:
            stamp = _parse_time(point.get("from"))
            if stamp is None:
                continue
            slot = (stamp - start) // 3600
            if 0 <= slot < 24:
                viewers = _number(point.get("viewers"))
                if viewers is not None and viewers > 0:
                    by_hour[slot] = viewers
        if len(by_hour) != 24:
            continue
        first = by_hour[0]
        log_first = math.log(first)
        ratios = [math.log(by_hour[h]) - log_first for h in range(1, 24)]
        pool.append((first, ratios))
    return case_id, p1, pool


def _wmean(values, weights):
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total


def _wstd(values, weights, centre):
    total = sum(weights)
    var = sum(w * (v - centre) ** 2 for v, w in zip(values, weights)) / total
    return math.sqrt(var) if var > 0.0 else 0.0


def _trimmed_wmean(values, weights, trim):
    """Weighted mean after dropping the lowest and highest `trim` share of
    total weight. Drops whole points from each tail until the trimmed weight
    is reached; with too few points left, falls back to the plain mean."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    total = sum(weights)
    keep = [True] * len(values)
    low = 0
    trimmed_low = 0.0
    while low < len(order) and trimmed_low < trim * total:
        keep[order[low]] = False
        trimmed_low += weights[order[low]]
        low += 1
    high = len(order) - 1
    trimmed_high = 0.0
    while high >= low and trimmed_high < trim * total:
        keep[order[high]] = False
        trimmed_high += weights[order[high]]
        high -= 1
    kept_values = [values[i] for i in order if keep[i]]
    kept_weights = [weights[i] for i in order if keep[i]]
    if len(kept_values) < 5:
        return _wmean(values, weights)
    return _wmean(kept_values, kept_weights)


def _ols_predict(values, xs, x_target):
    """Ordinary least squares of values on xs, predicted at x_target.
    Returns the fitted mean when the spread of xs is degenerate."""
    n = len(values)
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (v - y_mean) for x, v in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den < 1e-12:
        return y_mean
    slope = num / den
    return y_mean + slope * (x_target - x_mean)


def _smooth(values):
    """Three-point moving average; ends use the two available points."""
    out = []
    for i in range(len(values)):
        lo = i - 1 if i > 0 else 0
        hi = i + 2 if i + 2 <= len(values) else len(values)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _persistence(p1):
    """House baseline shape: every hour stays at the hour-1 print, with a
    spread widening over the day. Used when a case cannot be forecast from
    comparables at all."""
    rows = []
    for hour in range(2, 25):
        spread = 0.15 * math.sqrt(hour - 1)
        rows.append([p1 * math.exp(z * spread) for z in Z])
    return rows


def forecast(p1, pool):
    """Forecast 23 rows x 5 quantiles for one case. Falls back to the
    persistence shape when the pool is empty."""
    if not pool:
        return _persistence(p1)
    log_p1 = math.log(p1)
    log_firsts = [math.log(first) for first, _ in pool]
    centres = []
    sigmas = []
    for hour in range(23):
        values = [ratios[hour] for _, ratios in pool]
        weights = [math.exp(-0.5 * ((x - log_p1) / BANDWIDTH) ** 2)
                   for x in log_firsts]
        if sum(weights) <= 0.0:
            # Hour-1 print far outside the comparables' range: every
            # Gaussian weight underflowed. Fall back to equal weights so
            # the forecast degrades to the pooled mean, not to a crash.
            weights = [1.0] * len(weights)
        centre = _trimmed_wmean(values, weights, TRIM)
        ols = _ols_predict(values, log_firsts, log_p1)
        centre = (1.0 - OLS_BLEND) * centre + OLS_BLEND * ols
        sigma = _wstd(values, weights, centre)
        sigma = max(SIGMA_FLOOR, sigma) * SIGMA_SCALE
        centres.append(centre)
        sigmas.append(sigma)
    centres = _smooth(centres)
    sigmas = _smooth(sigmas)
    rows = []
    for hour in range(23):
        rows.append([p1 * math.exp(centres[hour] + z * sigmas[hour])
                     for z in Z])
    return rows


def _clean(rows):
    """Enforce the output contract: finite, above 0, at most 1e9,
    non-decreasing within each row."""
    clean = []
    for row in rows:
        fixed = []
        prev = 0.0
        for value in row:
            if value != value or value in (math.inf, -math.inf):
                value = prev if prev > 0 else 1.0
            value = min(max(value, 1.0), MAX_VIEWERS)
            if value < prev:
                value = prev
            fixed.append(value)
            prev = value
        clean.append(fixed)
    return clean


def run_case(input_path, output_dir):
    """Forecast one case file and write its output file. Returns the case
    id. Any failure inside the forecast falls back to persistence for that
    case, so this never raises for a readable input."""
    case_id, p1, pool = load_case(input_path)
    try:
        rows = forecast(p1, pool)
    except Exception:
        rows = _persistence(p1)
    rows = _clean(rows)
    hours = [{"hour": hour + 2, "q": rows[hour]} for hour in range(23)]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"hours": hours}, handle)
        handle.write("\n")
    return case_id


def main(argv):
    input_dir = "/input"
    output_dir = "/output"
    if len(argv) > 1:
        input_dir = argv[1]
    if len(argv) > 2:
        output_dir = argv[2]
    try:
        names = sorted(os.listdir(input_dir))
    except OSError:
        names = []
    inputs = [n for n in names if n.endswith(".in")]
    if not inputs:
        # Nothing to forecast: still write nothing and exit cleanly; the
        # runner scores a missing case as persistence, not as a crash.
        return 0
    for name in inputs:
        try:
            run_case(os.path.join(input_dir, name), output_dir)
        except Exception:
            # Last-resort guard: a case that cannot even be read still gets
            # a persistence forecast at the hour-1 print if we can find one.
            try:
                case_id = os.path.splitext(name)[0]
                with open(os.path.join(input_dir, name),
                          encoding="utf-8") as handle:
                    case = json.load(handle)
                known = case.get("known") or []
                p1 = _number(known[0].get("viewers")) if known else None
                if p1 is None or p1 <= 0:
                    p1 = 1.0
                rows = _clean(_persistence(p1))
                hours = [{"hour": h + 2, "q": rows[h]} for h in range(23)]
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, case_id + ".out"),
                          "w", encoding="utf-8") as handle:
                    json.dump({"hours": hours}, handle)
                    handle.write("\n")
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
