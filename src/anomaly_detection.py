import numpy as np
import pandas as pd


def detect_profile_shape_anomaly(df, tipo_giorno="feriale", n_reference_months=6, similarity_threshold=0.90):
    """Anomalia 1: cambiamento strutturale del profilo orario giornaliero."""
    sub = df[df["tipo_giorno"] == tipo_giorno].copy() 
    sub["data"] = pd.to_datetime(sub["data"])
    sub["year_month"] = sub["data"].dt.to_period("M")

    peak_hours = list(range(7, 10)) + list(range(17, 20))

    results = []

    for chiave, grp in sub.groupby("chiave"):
        monthly = (
            grp.groupby(["year_month", "ora_del_giorno"])["conteggio_veicoli"]
            .mean()
            .unstack("ora_del_giorno")
            .sort_index()
        )

        if len(monthly) <= n_reference_months:
            continue

        # normalize rows to unit sum to capture shape, not volume 
        profiles = monthly.div(monthly.sum(axis=1), axis=0)
        ref_profile = profiles.iloc[:n_reference_months].mean()
        ref_peak_share = ref_profile[peak_hours].sum() / ref_profile.sum()

        for month, row in profiles.iterrows():
            cos_sim = np.dot(ref_profile.values, row.values) / (
                np.linalg.norm(ref_profile.values) * np.linalg.norm(row.values) + 1e-9
            )
            peak_share = row[peak_hours].sum() / row.sum()
            peak_share_change = peak_share - ref_peak_share
            anomaly_score = (1 - cos_sim) + max(0, -peak_share_change)

            results.append({
                "chiave": chiave,
                "year_month": month,
                "cosine_sim": round(cos_sim, 4),
                "peak_share_ref": round(ref_peak_share, 4),
                "peak_share": round(peak_share, 4),
                "peak_share_change": round(peak_share_change, 4),
                "anomaly_score": round(anomaly_score, 4),
                "is_anomaly": cos_sim < similarity_threshold, #default 0.90
            })

    return pd.DataFrame(results)


def detect_trend_anomaly(df, reference_end=None, zscore_threshold=2.0, min_consecutive_months=2):
    """Anomalia 2: variazione persistente del flusso medio non spiegata dalla stagionalità."""
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df["year_month"] = df["data"].dt.to_period("M")
    df["month_of_year"] = df["data"].dt.month

    # remove event days and holidays
    clean = df[df["tipo_evento"].isna() & ~df["festivo_nazionale"]] #drop event days and holidays

    all_months = sorted(df["year_month"].unique())
    if reference_end:
        ref_end = pd.Period(reference_end, "M")
    else:
        ref_end = all_months[min(11, len(all_months) - 2)]

    # daily totals per sensor and day type
    daily = clean.groupby(["chiave", "data", "tipo_giorno", "year_month", "month_of_year"])["conteggio_veicoli"].sum().reset_index()

    ref_daily = daily[daily["year_month"] <= ref_end]
    test_daily = daily[daily["year_month"] > ref_end]

    # seasonal baseline from reference period
    baseline = ref_daily.groupby(["chiave", "month_of_year", "tipo_giorno"])["conteggio_veicoli"].agg(
        expected_mean="mean", expected_std="std", n_ref="count"
    ).reset_index()

    # monthly averages in test period
    monthly_test = test_daily.groupby(["chiave", "year_month", "tipo_giorno", "month_of_year"])["conteggio_veicoli"].mean().rename("actual_mean").reset_index()

    merged = monthly_test.merge(baseline, on=["chiave", "month_of_year", "tipo_giorno"], how="left")
    merged["pct_deviation"] = (merged["actual_mean"] - merged["expected_mean"]) / merged["expected_mean"] * 100

    se = merged["expected_std"] / np.sqrt(merged["n_ref"].fillna(1).clip(lower=1))
    merged["zscore"] = (merged["actual_mean"] - merged["expected_mean"]) / (se + 1e-9)
    merged["is_local_anomaly"] = merged["zscore"].abs() > zscore_threshold

    rows = []
    for (chiave, tipo_g), grp in merged.groupby(["chiave", "tipo_giorno"]):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        streak = 0
        for _, r in grp.iterrows():
            streak = streak + 1 if r["is_local_anomaly"] else 0
            rows.append({
                "chiave": r["chiave"],
                "year_month": r["year_month"],
                "tipo_giorno": r["tipo_giorno"],
                "actual_mean": round(r["actual_mean"], 1),
                "expected_mean": round(r["expected_mean"], 1),
                "pct_deviation": round(r["pct_deviation"], 2),
                "zscore": round(r["zscore"], 3),
                "consecutive_anomalous": streak,
                "is_anomaly": streak >= min_consecutive_months,
            })

    return pd.DataFrame(rows).sort_values(["chiave", "tipo_giorno", "year_month"])


def _get_event_family(event_name):
    """Remove trailing year from event name to group editions together."""
    parts = str(event_name).split()
    if parts and parts[-1].isdigit() and len(parts[-1]) == 4:
        return " ".join(parts[:-1])
    return str(event_name)


def detect_event_response_anomaly(df, zscore_threshold=2.0, min_editions=2):
    """Anomalia 3: risposta anomala del traffico a eventi ricorrenti e prevedibili."""
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])

    # only clean event days (no overlapping events)
    event_df = df[df["tipo_evento"].notna() & (df["eventi_sovrapposti"] == 0)].copy()
    event_df["event_family"] = event_df["evento_traffico"].map(_get_event_family)
    event_df["edition"] = event_df["evento_traffico"] + " [" + event_df["data"].dt.year.astype(str) + "]"

    results = []

    for chiave, grp_c in event_df.groupby("chiave"):
        grp_c = grp_c.copy()

        for (tipo_ev, fam), grp_f in grp_c.groupby(["tipo_evento", "event_family"]):
            editions = (
                grp_f.groupby("edition")
                .agg(
                    start_date=("data", "min"),
                    event_dates=("data_str", lambda s: set(s.unique())),
                    event_mean=("conteggio_veicoli", "mean"),
                    months=("data", lambda s: s.dt.month.unique().tolist()),
                )
                .sort_values("start_date")
                .reset_index()
            )

            if len(editions) < min_editions:
                continue

            # compute impact ratio for each edition vs same-month baseline
            impacts = []
            for _, ed in editions.iterrows():
                baseline_mask = (
                    (df["chiave"] == chiave)
                    & (df["data"].dt.month.isin(ed["months"]))
                    & (~df["data_str"].isin(ed["event_dates"]))
                    & (df["tipo_evento"].isna())
                    & (~df["festivo_nazionale"])
                )
                baseline_mean = df.loc[baseline_mask, "conteggio_veicoli"].mean()
                impact = (ed["event_mean"] - baseline_mean) / (baseline_mean + 1e-9)
                impacts.append(float(impact))

            # check if each edition is anomalous vs previous editions
            for i, (_, ed) in enumerate(editions.iterrows()):
                if i == 0:
                    results.append({
                        "chiave": chiave,
                        "tipo_evento": tipo_ev,
                        "event_family": fam,
                        "edition": ed["edition"],
                        "start_date": ed["start_date"],
                        "impact_ratio": round(impacts[i], 4),
                        "historical_mean_impact": np.nan,
                        "historical_std_impact": np.nan,
                        "zscore": np.nan,
                        "is_anomaly": False,
                    })
                    continue

                hist = impacts[:i]
                h_mean = float(np.mean(hist))
                h_std = float(np.std(hist)) if len(hist) > 1 else abs(h_mean) * 0.1
                h_std = max(h_std, 1e-3)

                zscore = (impacts[i] - h_mean) / h_std

                results.append({
                    "chiave": chiave,
                    "tipo_evento": tipo_ev,
                    "event_family": fam,
                    "edition": ed["edition"],
                    "start_date": ed["start_date"],
                    "impact_ratio": round(impacts[i], 4),
                    "historical_mean_impact": round(h_mean, 4),
                    "historical_std_impact": round(h_std, 4),
                    "zscore": round(zscore, 3),
                    "is_anomaly": abs(zscore) > zscore_threshold,
                })

    return pd.DataFrame(results)
