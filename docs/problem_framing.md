# Problem Framing

## Context

Bologna's municipality operates a network of inductive loop detectors ("spire") that report
hourly vehicle counts per sensor and direction. Standard traffic monitoring flags instantaneous
outliers — a single noisy reading, a brief spike or drop. That is not the anomaly this project
is after.

## The problem

The goal is to detect **persistent, structural deviations** from a sensor's expected behaviour,
as opposed to one-off noise:

- a **change in the daily traffic profile shape** (e.g. the morning/evening rush-hour pattern
  flattens or shifts, independent of overall volume),
- a **long-term trend shift** that is not explained by seasonality (e.g. a sensor reads
  persistently lower or higher than its seasonal baseline for multiple months — often a sign
  of road works, a permanent traffic re-routing, or a failing detector),
- an **abnormal response to a recurring, predictable event** (concerts, fairs, football
  matches), where the traffic impact of a given edition is a statistical outlier compared to
  previous editions of the same event.

A reading of 0 vehicles/hour is ambiguous on its own: it can mean a genuinely closed or empty
road, or a broken detector. This is why sensor **accuracy** (a per-hour data-quality score) is
treated as a first-class input rather than a technical detail — it lets the system tell those
two cases apart and avoid flagging detector failures as traffic anomalies.

## Data used to frame the problem

Hourly vehicle counts are enriched with weather (temperature, precipitation, wind), calendar
context (day of week, national/local holidays, weekday vs. weekend vs. holiday), and a
manually curated calendar of high-impact recurring events (football matches, trade fairs),
so that expected sources of variation (rain, holidays, known events) can be separated from
genuinely unexplained deviations. See the [README](../README.md) for the full data schema and
the three detectors that implement this framing (`src/anomaly_detection.py`), plus the ML
baselines and forecasting-based early-warning approach explored on top of them
(`src/baselines.py`, `notebooks/04_evaluation.ipynb`).
