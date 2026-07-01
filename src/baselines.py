import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


FEATURE_COLS = [
    "hour", "dow",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h",
    "temperature_2m", "precipitation", "wind_speed_10m",
]

PARAM_GRID = {
    "num_leaves":        [31, 63, 127],
    "learning_rate":     [0.05, 0.1],
    "min_child_samples": [20, 50, 100],
    "feature_fraction":  [0.7, 0.9],
    "reg_lambda":        [0.0, 0.1, 1.0],
}


def _add_season(df):
    month = df["data"].dt.month
    conditions = [
        month.isin([12, 1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8]),
        month.isin([9, 10, 11]),
    ]
    df["season"] = np.select(conditions, ["Winter", "Spring", "Summer", "Autumn"], default="")
    return df


def stat_analysis(df):
    df = df.copy()
    if "season" not in df.columns:
        df = _add_season(df)

    meteo_effect = (
        df.groupby("tempo")["conteggio_veicoli"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(meteo_effect["tempo"], meteo_effect["conteggio_veicoli"], color="steelblue")
    ax.set_title("Average Hourly Traffic by Weather Condition")
    ax.set_xlabel("Weather")
    ax.set_ylabel("Avg vehicles / hour")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    rain_profile = (
        df.groupby(["pioggia", "ora_del_giorno"])["conteggio_veicoli"]
          .mean()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    for rain, label, ls in [(False, "No rain", "-"), (True, "Rain", "--")]:
        sub = rain_profile[rain_profile["pioggia"] == rain]
        ax.plot(sub["ora_del_giorno"], sub["conteggio_veicoli"],
                linestyle=ls, linewidth=2, label=label)
    ax.set_title("Hourly Traffic Profile: Rain vs No Rain")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg vehicles / hour")
    ax.set_xticks(range(24))
    ax.legend()
    plt.tight_layout()
    plt.show()

    rain_season = (
        df.groupby(["season", "pioggia"])["conteggio_veicoli"]
          .mean()
          .unstack("pioggia")
          .rename(columns={False: "No rain", True: "Rain"})
          .reindex(["Winter", "Spring", "Summer", "Autumn"])
    )
    rain_season["delta_pct"] = (
        (rain_season["Rain"] - rain_season["No rain"]) / rain_season["No rain"] * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    rain_season[["No rain", "Rain"]].plot(kind="bar", ax=axes[0],
                                          color=["steelblue", "slategray"])
    axes[0].set_title("Avg Traffic: Rain vs No Rain by Season")
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Avg vehicles / hour")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].bar(rain_season.index, rain_season["delta_pct"],
                color=["#d62728" if v < 0 else "#2ca02c" for v in rain_season["delta_pct"]])
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Rain Effect on Traffic (% change vs no-rain)")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Δ% vehicles / hour")
    axes[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()

    df = df.copy()
    df["precip_bin"] = pd.cut(
        df["precipitation"],
        bins=[-0.01, 0, 1, 5, 10, df["precipitation"].max() + 1],
        labels=["0 mm", "0–1 mm", "1–5 mm", "5–10 mm", ">10 mm"],
    )
    precip_effect = (
        df.groupby("precip_bin", observed=True)["conteggio_veicoli"]
          .agg(mean="mean", count="count")
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(precip_effect["precip_bin"].astype(str),
                  precip_effect["mean"], color="steelblue")
    ax.set_title("Avg Traffic by Precipitation Intensity")
    ax.set_xlabel("Precipitation (mm/h)")
    ax.set_ylabel("Avg vehicles / hour")
    for bar, cnt in zip(bars, precip_effect["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"n={cnt:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()

    weather_corr = (
        df[["conteggio_veicoli", "temperature_2m", "precipitation",
            "wind_speed_10m", "pioggia"]]
        .assign(pioggia=df["pioggia"].astype(int))
        .corr()["conteggio_veicoli"]
        .drop("conteggio_veicoli")
        .sort_values()
    )
    print("Pearson correlation with conteggio_veicoli:")
    print(weather_corr.to_string())

    return weather_corr, precip_effect, df


def feat_engineering(df):
    df = df.copy()
    df["hour"] = df["data"].dt.hour
    df["dow"] = df["data"].dt.dayofweek
    g = df.groupby("chiave")["conteggio_veicoli"]
    df["lag_1h"] = g.shift(1)
    df["lag_24h"] = g.shift(24)
    df["lag_168h"] = g.shift(168)
    df["rolling_mean_3h"] = g.transform(lambda x: x.rolling(window=3).mean())
    df["rolling_mean_24h"] = g.transform(lambda x: x.rolling(window=24).mean())

    for col in ["nome_festivita", "evento_traffico", "tipo_evento", "impatto_evento"]:
        df[col] = df[col].fillna("missing").astype("category")

    for col in ["nome_via", "direzione", "fascia_oraria", "tempo", "giorno_settimana",
                "nome_festivita", "evento_traffico", "tipo_evento", "impatto_evento",
                "tipo_giorno", "weekday"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def split(df):
    train = df[df["data"] < "2025-01-01"]
    test  = df[df["data"] >= "2025-01-01"]
    return train, test


def isolation_forest(train, test):
    train_X = train[FEATURE_COLS].dropna()
    test_X  = test[FEATURE_COLS].dropna()

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(train_X)

    result = test.loc[test_X.index].copy()
    result["anomaly_score"]     = model.score_samples(test_X)
    result["anomaly_isolation"] = model.predict(test_X)
    return result


def local_outlier_factor(df):
    clean = df[FEATURE_COLS].dropna()

    model = LocalOutlierFactor(n_neighbors=20, contamination=0.01)

    result = df.loc[clean.index].copy()
    result["anomaly_lof"] = model.fit_predict(clean)
    result["lof_score"]   = model.negative_outlier_factor_
    return result


def prepare_lgb_dataset(df):
    train, test = split(df)

    X_train = train[FEATURE_COLS]
    y_train = train["conteggio_veicoli"]

    X_test = test[FEATURE_COLS]
    y_test = test["conteggio_veicoli"]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval  = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "boosting_type":    "gbdt",
        "objective":        "regression",
        "metric":           ["l2", "l1"],
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          0,
    }

    gbm = lgb.train(
        params, lgb_train, num_boost_round=20,
        valid_sets=lgb_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
    )
    return gbm


def evaluation(gbm, X_test, y_test):
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    return mse, mae


def lgb_grid_search(df, n_splits=3):
    df = feat_engineering(df).sort_values("data")

    clean = df[FEATURE_COLS + ["conteggio_veicoli"]].dropna()
    X, y = clean[FEATURE_COLS], clean["conteggio_veicoli"]

    base = lgb.LGBMRegressor(
        boosting_type="gbdt",
        n_estimators=300,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
    )

    gs = GridSearchCV(
        base,
        PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    gs.fit(X, y)

    print(f"\nBest MAE  : {-gs.best_score_:.2f}")
    print(f"Best params: {gs.best_params_}")
    return gs


def show_cv_results(gs, top_n=10):
    return (
        pd.DataFrame(gs.cv_results_)
          .sort_values("rank_test_score")
          .assign(
              mean_MAE=lambda d: (-d["mean_test_score"]).round(2),
              std_MAE=lambda d: d["std_test_score"].round(2),
          )
          [["rank_test_score", "mean_MAE", "std_MAE", "params"]]
          .head(top_n)
    )
