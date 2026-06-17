import re
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "detect_profile_shape_anomaly",
    "detect_trend_anomaly",
    "detect_event_response_anomaly",
]


# ---------------------------------------------------------------------------
# Anomaly 1 — structural change in the daily hourly traffic profile
# ---------------------------------------------------------------------------


def detect_profile_shape_anomaly(
    df: pd.DataFrame,
    tipo_giorno: str = "feriale",
    n_reference_months: int = 6,
    similarity_threshold: float = 0.90,
    morning_hours: tuple = (7, 9),
    evening_hours: tuple = (17, 19),
    min_days_per_month: int = 10,
) -> pd.DataFrame:
    """
    Anomalia 1: cambiamento strutturale del profilo orario giornaliero.

    Per ogni sensore costruisce un profilo orario di riferimento (media dei
    primi n_reference_months mesi, normalizzato a somma 1) e lo confronta
    mese per mese con i profili successivi.  Segnala i mesi in cui la forma
    del profilo si allontana significativamente: per esempio un asse viario
    che perde la classica doppia gobba mattina-sera.

    Metriche calcolate
    ------------------
    cosine_sim : similarità coseno tra il profilo del mese e il riferimento
                 (1 = identico, 0 = ortogonale).
    peak_share : frazione del flusso concentrata nelle fasce di punta
                 (morning_hours + evening_hours) sul profilo normalizzato.
    peak_share_change : variazione di peak_share rispetto al riferimento;
                        valori negativi indicano perdita di bimodalità.
    anomaly_score : (1 - cosine_sim) + max(0, -peak_share_change).
    is_anomaly : True se cosine_sim < similarity_threshold.

    Parameters
    ----------
    df : DataFrame pre-filtrato con colonne tipo_giorno, ora_del_giorno,
         conteggio_veicoli, chiave, data.
    tipo_giorno : tipo di giorno su cui calcolare il profilo.
    n_reference_months : mesi iniziali usati come baseline di forma.
    similarity_threshold : soglia coseno sotto cui il mese è anomalo.
    morning_hours : (inizio, fine) del picco mattutino, estremi inclusi.
    evening_hours : (inizio, fine) del picco serale, estremi inclusi.
    min_days_per_month : giorni validi minimi per includere un mese.

    Returns
    -------
    pd.DataFrame con colonne: chiave, year_month, cosine_sim,
    peak_share_ref, peak_share, peak_share_change, anomaly_score, is_anomaly.
    """
    sub = df[df["tipo_giorno"] == tipo_giorno].copy()
    sub["data"] = pd.to_datetime(sub["data"])
    sub["year_month"] = sub["data"].dt.to_period("M")

    peak_hours: set[int] = set(range(morning_hours[0], morning_hours[1] + 1)) | set(
        range(evening_hours[0], evening_hours[1] + 1)
    )

    def _peak_share(profile: pd.Series) -> float:
        cols = [h for h in peak_hours if h in profile.index]
        return float(profile[cols].sum() / (profile.sum() + 1e-9))

    results = []

    for chiave, grp in sub.groupby("chiave"):
        # keep only months with enough days
        days_pm = grp.groupby("year_month")["data"].nunique()
        valid = days_pm[days_pm >= min_days_per_month].index

        monthly_raw = (
            grp[grp["year_month"].isin(valid)]
            .groupby(["year_month", "ora_del_giorno"])["conteggio_veicoli"]
            .mean()
            .unstack("ora_del_giorno")
            .sort_index()
        )

        if len(monthly_raw) <= n_reference_months:
            continue

        # normalize rows to unit sum to captures shape, not volume
        profiles = monthly_raw.div(monthly_raw.sum(axis=1), axis=0)
        ref_profile = profiles.iloc[:n_reference_months].mean()
        ref_ps = _peak_share(ref_profile)
        ref_v = ref_profile.values

        for month, row in profiles.iterrows():
            row_v = row.values
            denom = np.linalg.norm(ref_v) * np.linalg.norm(row_v)
            cos_sim = float(np.dot(ref_v, row_v) / (denom + 1e-9))

            ps = _peak_share(row)
            ps_change = ps - ref_ps
            anomaly_score = (1.0 - cos_sim) + max(0.0, -ps_change)

            results.append(
                {
                    "chiave": chiave,
                    "year_month": month,
                    "cosine_sim": round(cos_sim, 4),
                    "peak_share_ref": round(ref_ps, 4),
                    "peak_share": round(ps, 4),
                    "peak_share_change": round(ps_change, 4),
                    "anomaly_score": round(anomaly_score, 4),
                    "is_anomaly": bool(cos_sim < similarity_threshold),
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Anomaly 2 — persistent flow-level shift unexplained by seasonality
# ---------------------------------------------------------------------------


def detect_trend_anomaly(
    df: pd.DataFrame,
    reference_end: Optional[str] = None,
    zscore_threshold: float = 2.0,
    min_consecutive_months: int = 2,
    exclude_events: bool = True,
) -> pd.DataFrame:
    """
    Anomalia 2: variazione persistente del flusso medio non spiegata dalla stagionalità.

    Divide i dati in un periodo di riferimento e un periodo di test.  Per ogni
    sensore e tipo di giorno costruisce un calendario stagionale atteso
    (media per mese dell'anno nel riferimento) e confronta i mesi successivi
    con quello stesso calendario.  Un mese è localmente anomalo se il suo
    z-score supera la soglia; diventa un'anomalia strutturale solo se l'eccesso
    persiste per almeno min_consecutive_months mesi consecutivi.

    Lo z-score è calcolato come
        (media_mensile_test − media_stagionale_riferimento) /
        (deviazione_standard_giornaliera_riferimento / √n_giorni_riferimento)
    e mette in relazione lo scostamento con la variabilità giornaliera tipica
    osservata nel periodo di calibrazione.

    Parameters
    ----------
    df : DataFrame pre-filtrato.
    reference_end : ultimo mese del periodo di riferimento (es. '2024-08').
                    Default: i primi 12 mesi del dataset.
    zscore_threshold : soglia |z| per classificare un mese come localmente anomalo.
    min_consecutive_months : mesi anomali consecutivi necessari per emettere alert.
    exclude_events : se True esclude giorni con eventi e festività nazionali.

    Returns
    -------
    pd.DataFrame con colonne: chiave, year_month, tipo_giorno, actual_mean,
    expected_mean, pct_deviation, zscore, consecutive_anomalous, is_anomaly.
    """
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df["year_month"] = df["data"].dt.to_period("M")
    df["month_of_year"] = df["data"].dt.month

    if exclude_events:
        clean = df[df["tipo_evento"].isna() & ~df["festivo_nazionale"]]
    else:
        clean = df

    all_months = sorted(df["year_month"].unique())
    ref_end = (
        pd.Period(reference_end, "M")
        if reference_end
        else all_months[min(11, len(all_months) - 2)]
    )

    # daily totals per sensor/tipo_giorno
    daily = (
        clean.groupby(["chiave", "data", "tipo_giorno", "year_month", "month_of_year"])[
            "conteggio_veicoli"
        ]
        .sum()
        .reset_index()
    )

    ref_daily = daily[daily["year_month"] <= ref_end]
    test_daily = daily[daily["year_month"] > ref_end]

    # seasonal baseline from reference: mean and std of daily totals
    baseline = (
        ref_daily.groupby(["chiave", "month_of_year", "tipo_giorno"])["conteggio_veicoli"]
        .agg(expected_mean="mean", expected_std="std", n_ref="count")
        .reset_index()
    )

    # monthly mean in the test period
    monthly_test = (
        test_daily.groupby(["chiave", "year_month", "tipo_giorno", "month_of_year"])[
            "conteggio_veicoli"
        ]
        .mean()
        .rename("actual_mean")
        .reset_index()
    )

    merged = monthly_test.merge(
        baseline, on=["chiave", "month_of_year", "tipo_giorno"], how="left"
    )

    merged["pct_deviation"] = (
        (merged["actual_mean"] - merged["expected_mean"])
        / (merged["expected_mean"] + 1e-9)
        * 100
    )
    # standard error of the reference seasonal mean
    se = merged["expected_std"] / np.sqrt(merged["n_ref"].fillna(1).clip(lower=1))
    merged["zscore"] = (merged["actual_mean"] - merged["expected_mean"]) / (se + 1e-9)
    merged["_anomaly_month"] = merged["zscore"].abs() > zscore_threshold

    # flag persistent runs
    rows = []
    for (chiave, tipo_g), grp in merged.groupby(["chiave", "tipo_giorno"]):
        grp = grp.sort_values("year_month").reset_index(drop=True)
        streak = 0
        for _, r in grp.iterrows():
            streak = (streak + 1) if r["_anomaly_month"] else 0
            rows.append(
                {
                    "chiave": r["chiave"],
                    "year_month": r["year_month"],
                    "tipo_giorno": r["tipo_giorno"],
                    "actual_mean": round(r["actual_mean"], 1),
                    "expected_mean": round(r["expected_mean"], 1),
                    "pct_deviation": round(r["pct_deviation"], 2),
                    "zscore": round(r["zscore"], 3),
                    "consecutive_anomalous": streak,
                    "is_anomaly": bool(streak >= min_consecutive_months),
                }
            )

    return pd.DataFrame(rows).sort_values(["chiave", "tipo_giorno", "year_month"])


# ---------------------------------------------------------------------------
# Anomaly 3 — anomalous traffic response to recurring predictable events
# ---------------------------------------------------------------------------


def _event_family(name: str) -> str:
    """Strip trailing 4-digit year from event name to get the recurring family."""
    return re.sub(r"[\s|]*\b\d{4}\b[\s|]*$", "", str(name)).strip()


def detect_event_response_anomaly(
    df: pd.DataFrame,
    zscore_threshold: float = 2.0,
    min_editions: int = 2,
) -> pd.DataFrame:
    """
    Anomalia 3: risposta anomala del traffico a eventi ricorrenti e prevedibili.

    Per ogni evento ricorrente (fiera, partita di calcio dello stesso tipo)
    calcola l'impatto sul flusso di ciascuna edizione come rapporto relativo
    tra il flusso medio durante l'evento e un flusso di riferimento del
    sensore nello stesso mese (giorni senza eventi e senza festività).

    L'edizione più recente è poi confrontata con la distribuzione degli
    impatti delle edizioni precedenti: se lo z-score supera la soglia
    l'anomalia segnala che la risposta dell'asse viario a quell'evento è
    cambiata rispetto alle edizioni passate (diverso instradamento,
    diversa attrattività, diversa risposta modale, ecc.).

    Vengono esclusi i giorni con eventi sovrapposti (eventi_sovrapposti > 0)
    per evitare effetti di confondimento.

    Parameters
    ----------
    df : DataFrame pre-filtrato.
    zscore_threshold : soglia |z| per classificare un'edizione come anomala.
    min_editions : numero minimo di edizioni necessarie per calcolare
                   la distribuzione storica (default 2).

    Returns
    -------
    pd.DataFrame con colonne: chiave, tipo_evento, event_family, edition,
    start_date, impact_ratio, historical_mean_impact, historical_std_impact,
    zscore, is_anomaly.

    Nota: la prima edizione di ciascuna famiglia ha zscore=NaN e
    is_anomaly=False perché non ha storia su cui basarsi.
    """
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])

    # solo eventi "puri" (nessun sovrapposizione)
    event_df = df[df["tipo_evento"].notna() & (df["eventi_sovrapposti"] == 0)].copy()
    event_df["event_family"] = event_df["evento_traffico"].map(_event_family)
    # un'edizione = stesso evento_traffico nello stesso anno solare
    event_df["edition"] = (
        event_df["evento_traffico"]
        + " ["
        + event_df["data"].dt.year.astype(str)
        + "]"
    )

    results = []

    for chiave, grp_c in event_df.groupby("chiave"):
        grp_c = grp_c.copy()
        grp_c["family_key"] = grp_c["tipo_evento"] + "|" + grp_c["event_family"]

        for family_key, grp_f in grp_c.groupby("family_key"):
            tipo_ev, fam = family_key.split("|", 1)

            # aggrega per edizione: data di inizio, date evento, media oraria, mesi coinvolti
            editions = (
                grp_f.groupby("edition")
                .agg(
                    start_date=("data", "min"),
                    event_dates_str=("data_str", lambda s: sorted(s.unique().tolist())),
                    event_mean=("conteggio_veicoli", "mean"),
                    months=("data", lambda s: s.dt.month.unique().tolist()),
                )
                .sort_values("start_date")
                .reset_index()
            )

            if len(editions) < min_editions:
                continue

            # calcola impatto di ciascuna edizione
            impacts: list[float] = []
            for _, ed in editions.iterrows():
                event_dates_set: set[str] = set(ed["event_dates_str"])
                baseline_mask = (
                    (df["chiave"] == chiave)
                    & (df["data"].dt.month.isin(ed["months"]))
                    & (~df["data_str"].isin(event_dates_set))
                    & (df["tipo_evento"].isna())
                    & (~df["festivo_nazionale"])
                )
                baseline_mean = df.loc[baseline_mask, "conteggio_veicoli"].mean()
                impact = float(
                    (ed["event_mean"] - baseline_mean) / (baseline_mean + 1e-9)
                )
                impacts.append(impact)

            # confronta ogni edizione con le precedenti
            for i, (_, ed) in enumerate(editions.iterrows()):
                if i == 0:
                    results.append(
                        {
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
                        }
                    )
                    continue

                hist = impacts[:i]
                h_mean = float(np.mean(hist))
                # con una sola edizione storica non si può stimare std:
                # si usa il 10 % del valore assoluto come stima conservativa
                h_std = float(np.std(hist, ddof=0)) if len(hist) > 1 else abs(h_mean) * 0.1
                h_std = max(h_std, 1e-3)  # avoid division by zero

                zscore = round((impacts[i] - h_mean) / h_std, 3)

                results.append(
                    {
                        "chiave": chiave,
                        "tipo_evento": tipo_ev,
                        "event_family": fam,
                        "edition": ed["edition"],
                        "start_date": ed["start_date"],
                        "impact_ratio": round(impacts[i], 4),
                        "historical_mean_impact": round(h_mean, 4),
                        "historical_std_impact": round(h_std, 4),
                        "zscore": zscore,
                        "is_anomaly": bool(abs(zscore) > zscore_threshold),
                    }
                )

    return pd.DataFrame(results)
