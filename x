import numpy as np
import pandas as pd


def make_stat_features(
    df,
    *,
    lags=(1, 2, 3, 7, 14, 21, 28, 30, 56, 90, 365),
    windows=(7, 14, 28, 56, 90),
    min_coverage=0.5,
):
    """Добавляет к REP_DT и OST_VAL признаки по истории одного ряда."""
    if any(not isinstance(lag, (int, np.integer)) or lag < 1 for lag in lags):
        raise ValueError("Лаги должны быть положительными целыми числами.")
    if any(not isinstance(window, (int, np.integer)) or window < 2 for window in windows):
        raise ValueError("Размеры окон должны быть целыми числами от 2.")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage должен быть от 0 до 1.")

    dates = pd.DatetimeIndex(pd.to_datetime(df["REP_DT"], format="mixed"))
    dates = dates.tz_localize(None).normalize()
    if dates.hasnans or dates.has_duplicates:
        raise ValueError("Нужна одна строка на дату, без пропусков в REP_DT.")

    base = df.assign(REP_DT=dates).set_index("REP_DT").sort_index()
    calendar = (
        pd.date_range(base.index[0], base.index[-1], freq="D", name="REP_DT")
        if len(base) else base.index
    )
    balance = (
        pd.to_numeric(base["OST_VAL"], errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .reindex(calendar)
    )

    # Для даты t история заканчивается вчера. Лаги считаются в календарных днях.
    past = balance.shift(1)
    past_change = past.diff()
    features = {}

    def safe_ratio(numerator, denominator):
        return numerator.div(denominator.abs().where(denominator.ne(0)))

    def slope_per_day(values):
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.nan
        x = np.flatnonzero(valid)
        y = values[valid]
        x = x - x.mean()
        return (x @ (y - y.mean())) / (x @ x)

    for lag in lags:
        delayed = balance.shift(lag)
        features[f"stat_lag_{lag}d"] = delayed
        if lag > 1:
            delta = past - delayed
            features[f"stat_delta_lag1_vs_lag{lag}d"] = delta
            features[f"stat_rel_delta_lag1_vs_lag{lag}d"] = safe_ratio(delta, delayed)

    date_series = pd.Series(calendar, index=calendar)
    last_observed = date_series.where(balance.notna()).shift(1).ffill()
    features["stat_days_since_observation"] = (date_series - last_observed).dt.days

    for days in windows:
        minimum = max(1, int(np.ceil(days * min_coverage)))
        rolling = past.rolling(days, min_periods=minimum)
        changes = past_change.rolling(days, min_periods=minimum)
        mean, std = rolling.mean(), rolling.std()
        low, high = rolling.min(), rolling.max()
        q25, q75 = rolling.quantile(0.25), rolling.quantile(0.75)
        count = past.notna().rolling(days, min_periods=1).sum()

        stats = {
            "count": count,
            "coverage": count / days,
            "mean": mean,
            "median": rolling.median(),
            "std": std,
            "min": low,
            "max": high,
            "range": high - low,
            "q10": rolling.quantile(0.10),
            "q25": q25,
            "q75": q75,
            "q90": rolling.quantile(0.90),
            "iqr": q75 - q25,
            "skew": rolling.skew(),
            "slope": rolling.apply(slope_per_day, raw=True),
            "lag1_minus_mean": past - mean,
            "lag1_zscore": safe_ratio(past - mean, std),
            "change_mean": changes.mean(),
            "change_std": changes.std(),
            "change_abs_mean": past_change.abs().rolling(days, min_periods=minimum).mean(),
            "change_up_share": (
                past_change.gt(0).where(past_change.notna())
                .rolling(days, min_periods=minimum).mean()
            ),
        }
        features.update({f"stat_roll_{days}d_{name}": values for name, values in stats.items()})

    for span in (7, 28, 90):
        ew = past.ewm(span=span, adjust=False, min_periods=2, ignore_na=False)
        features[f"stat_ewm_{span}d_mean"] = ew.mean()
        features[f"stat_ewm_{span}d_std"] = ew.std()

    for weeks in (4, 8):
        same_weekday = pd.concat(
            [balance.shift(7 * week) for week in range(1, weeks + 1)], axis=1
        )
        count = same_weekday.count(axis=1)
        enough = count.ge(max(1, int(np.ceil(weeks * min_coverage))))
        prefix = f"stat_same_weekday_{weeks}w"
        features[f"{prefix}_count"] = count
        features[f"{prefix}_mean"] = same_weekday.mean(axis=1).where(enough)
        features[f"{prefix}_median"] = same_weekday.median(axis=1).where(enough)
        features[f"{prefix}_std"] = same_weekday.std(axis=1).where(enough)

    stat = pd.DataFrame(features, index=calendar).replace([np.inf, -np.inf], np.nan)
    # Возвращаем исходные отчётные даты. Пропуски внутри истории сохраняются.
    return base.join(stat.reindex(base.index)).reset_index()


df_stat = make_stat_features(df_fl)
df_stat.tail()
