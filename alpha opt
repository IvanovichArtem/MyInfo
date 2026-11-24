"""Дневная модель остатков и бэктест на последнем календарном годе.

Вход: один ряд df с колонками date и amount, по одному остатку в день.
Остатки неотрицательны, денежная единица произвольна. Нужны numpy и pandas.
"""

import numpy as np
import pandas as pd


AR_LAGS = (1, 7)
DEFAULT_HORIZONS = (1, 7, 30, 90, 180, 365)
MIN_HISTORY = max(AR_LAGS) + 3  # Две или более ошибки после построения лагов.


def _prepare(df):
    if not {"date", "amount"}.issubset(df.columns):
        raise ValueError("В df нужны столбцы date и amount.")

    dates = pd.to_datetime(df["date"], errors="raise").dt.tz_localize(None)
    amounts = pd.to_numeric(df["amount"], errors="raise").to_numpy(dtype=float)
    if dates.isna().any() or not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Нужны даты без пропусков и конечные неотрицательные остатки.")

    series = pd.Series(amounts, index=pd.DatetimeIndex(dates.dt.normalize(), name="date"))
    series = series.sort_index()
    if not series.index.is_unique:
        raise ValueError("По каждому календарному дню нужен один остаток.")
    if len(series) < MIN_HISTORY:
        raise ValueError(f"Для оценки модели нужны минимум {MIN_HISTORY} дневных остатков.")
    expected = pd.date_range(series.index[0], series.index[-1], freq="D", name="date")
    if not series.index.equals(expected):
        raise ValueError("В ряду есть пропущенные календарные дни. Проверьте историю.")
    return series


def _calendar_features(dates):
    weekday = dates.dayofweek.to_numpy()
    phase = 2 * np.pi * (dates.day.to_numpy() - 1) / dates.days_in_month.to_numpy()
    month_end = dates.days_in_month.to_numpy() - dates.day.to_numpy() < 3
    return np.column_stack((
        weekday[:, None] == np.arange(6),
        np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase),
        month_end,
    ))


def _fit_arx(changes, dates):
    positions = np.arange(max(AR_LAGS), len(changes))
    target = changes[positions]
    design = np.column_stack((
        np.ones(len(positions)),
        changes[positions[:, None] - np.array(AR_LAGS)],
        _calendar_features(dates[positions]),
    ))

    # Ridge с единичным штрафом. Свободный член не штрафуется.
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    coefficients = np.linalg.lstsq(
        np.vstack((design, penalty)),
        np.concatenate((target, np.zeros(len(penalty)))),
        rcond=None,
    )[0]

    # Достаточное условие устойчивости рекурсивной части.
    ar = coefficients[1:1 + len(AR_LAGS)]
    strength = np.abs(ar).sum()
    if strength > 0.98:
        ar *= 0.98 / strength
        coefficients[0] += (target - design @ coefficients).mean()

    residuals = target - design @ coefficients
    return coefficients, residuals - residuals.mean()


def _simulate(level, recent, volatility, date_effect, ar, innovations):
    steps, n_paths = innovations.shape
    lag_days = np.array(AR_LAGS)
    history = np.broadcast_to(recent[:, None], (len(recent), n_paths)).copy()
    current = np.full(n_paths, level)
    paths = np.empty_like(innovations)

    for step in range(steps):
        lag_values = history[(step - lag_days) % len(recent)]
        change = date_effect[step] + ar @ lag_values + innovations[step]
        following = np.maximum(0.0, current + change * volatility)
        history[step % len(recent)] = (following - current) / volatility
        paths[step] = following
        current = following
    return paths


def _forecast(history, dates, n_paths, q, rng):
    balances = history.to_numpy()
    start_amount = balances[-1]
    positive_balances = balances[balances > 0]
    scale = float(np.median(positive_balances)) if positive_balances.size else 1.0
    levels = np.log1p(balances / scale)
    changes = np.diff(levels)
    volatility = float(changes.std()) or 1.0
    changes /= volatility

    coefficients, residuals = _fit_arx(changes, history.index[1:])
    ar_end = 1 + len(AR_LAGS)
    calendar = coefficients[0] + _calendar_features(dates) @ coefficients[ar_end:]
    recent = changes[-max(AR_LAGS):]
    models = (
        ("arx", calendar, coefficients[1:ar_end], residuals),
        ("naive", np.zeros(len(dates)), np.zeros(len(AR_LAGS)), changes - changes.mean()),
    )

    results = {}
    for name, date_effect, ar, errors in models:
        innovations = rng.choice(errors, size=(len(dates), n_paths))
        paths = _simulate(levels[-1], recent, volatility, date_effect, ar, innovations)
        paths = scale * np.expm1(paths)
        drawdowns = np.maximum.accumulate(np.maximum(start_amount - paths, 0.0), axis=0)
        lower, median, upper = np.quantile(paths, (1 - q, 0.5, q), axis=1)
        drawdown_q = np.quantile(drawdowns, q, axis=1)
        results[name] = pd.DataFrame({
            "origin_date": history.index[-1],
            "date": dates,
            "model": name,
            "horizon_days": np.arange(1, len(dates) + 1),
            "start_amount": start_amount,
            "forecast_amount": median if name == "arx" else start_amount,
            "amount_lower": lower,
            "amount_upper": upper,
            "forecast_max_drawdown_q": drawdown_q,
            "forecast_min_amount": start_amount - drawdown_q,
        })
    return pd.concat(results.values(), ignore_index=True)


def _positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0


def _check_simulation(n_paths, q):
    if not _positive_integer(n_paths) or n_paths < 2:
        raise ValueError("n_paths должно быть целым числом от 2.")
    if not 0.5 < q < 1:
        raise ValueError("q должен быть между 0.5 и 1, например 0.9.")


def forecast_ump(df, *, days=365, n_paths=1000, q=0.90, seed=42):
    """Прогноз после последней даты ряда, одна строка на модель и день вперёд."""
    _check_simulation(n_paths, q)
    if not _positive_integer(days):
        raise ValueError("days должно быть положительным целым числом.")
    series = _prepare(df)
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    return _forecast(series, dates, n_paths, q, np.random.default_rng(seed))


def _with_actual(forecast, actual):
    horizon_indices = forecast["horizon_days"].to_numpy() - 1
    actual_at_horizon = actual[horizon_indices]
    start = forecast["start_amount"].to_numpy()
    actual_drawdown = np.maximum(start - np.minimum.accumulate(actual)[horizon_indices], 0.0)
    return forecast.assign(actual_amount=actual_at_horizon, actual_max_drawdown=actual_drawdown)


def _level_metrics(rows, mase_scale):
    actual = rows["actual_amount"].to_numpy()
    predicted = rows["forecast_amount"].to_numpy()
    error = predicted - actual
    absolute_error = np.abs(error)
    smape = np.divide(2 * absolute_error, actual + predicted, out=np.zeros_like(error),
                      where=actual + predicted > 0)
    within_interval = (
        (rows["actual_amount"] >= rows["amount_lower"])
        & (rows["actual_amount"] <= rows["amount_upper"])
    )
    return {
        "n": len(rows),
        "mae": absolute_error.mean(),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "smape_pct": 100 * smape.mean(),
        "mase": absolute_error.mean() / mase_scale if mase_scale else np.nan,
        "bias": error.mean(),
        "balance_interval_coverage_pct": 100 * within_interval.mean(),
    }


def _drawdown_errors(rows, q):
    error = rows["actual_max_drawdown"] - rows["forecast_max_drawdown_q"]
    excess = error.clip(lower=0)
    return rows.assign(
        drawdown_pinball=np.where(error >= 0, q * error, (q - 1) * error),
        drawdown_exceeded=error > 0,
        drawdown_excess=excess,
    )


def backtest_ump(df, *, horizons=DEFAULT_HORIZONS, stride=7, n_paths=1000, q=0.90, seed=42):
    """Вернуть фиксированный прогноз на год и метрики rolling-origin по горизонтам.

    Последний календарный год служит тестом. Фиксированный прогноз использует
    только историю до начала теста. Rolling-origin добавляет на каждой дате
    только уже наблюдавшиеся остатки. Горизонты измеряются в календарных днях.
    """
    _check_simulation(n_paths, q)
    horizons = tuple(horizons)
    if not _positive_integer(stride) or not horizons or not all(map(_positive_integer, horizons)):
        raise ValueError("stride и каждый из horizons должны быть положительными целыми числами.")

    series = _prepare(df)
    cutoff = series.index[-1] - pd.DateOffset(years=1)
    train_size = series.index.searchsorted(cutoff, side="right")
    if train_size < MIN_HISTORY:
        raise ValueError(f"До последнего календарного года нужны минимум {MIN_HISTORY} наблюдений.")

    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    selected = sorted({h for h in horizons if h <= len(test)})
    if not selected:
        raise ValueError("Ни один из заданных горизонтов не помещается в тестовый год.")

    mase_scale = np.abs(np.diff(train.to_numpy())).mean()
    rng = np.random.default_rng(seed)
    annual = _forecast(train, test.index, n_paths, q, rng)
    annual = _drawdown_errors(_with_actual(annual, test.to_numpy()), q)

    rolling = [annual.loc[annual["horizon_days"].isin(selected)]]
    for offset in range(stride, len(test) - selected[0] + 1, stride):
        available = [h for h in selected if h <= len(test) - offset]
        origin = train_size + offset
        future = series.iloc[origin:origin + max(available)]
        forecast = _forecast(series.iloc[:origin], future.index, n_paths, q, rng)
        rows = forecast.loc[forecast["horizon_days"].isin(available)]
        rolling.append(_drawdown_errors(_with_actual(rows, future.to_numpy()), q))

    rolling = pd.concat(rolling, ignore_index=True)
    horizon_metrics = []
    for (model, horizon), rows in rolling.groupby(["model", "horizon_days"], sort=True):
        horizon_metrics.append({
            "model": model,
            "horizon_days": horizon,
            **_level_metrics(rows, mase_scale),
            "drawdown_q": q,
            "drawdown_pinball": rows["drawdown_pinball"].mean(),
            "drawdown_breach_pct": 100 * rows["drawdown_exceeded"].mean(),
            "mean_drawdown_excess": rows["drawdown_excess"].mean(),
        })

    annual_drawdown_cols = [
        "model", "origin_date", "date", "horizon_days", "start_amount", "actual_max_drawdown",
        "forecast_max_drawdown_q", "forecast_min_amount", "drawdown_pinball", "drawdown_exceeded",
        "drawdown_excess",
    ]
    return {
        "annual_forecast": annual,
        "annual_metrics": pd.DataFrame([
            {"model": model, **_level_metrics(rows, mase_scale)}
            for model, rows in annual.groupby("model", sort=True)
        ]),
        "annual_drawdown": annual.loc[annual["horizon_days"].eq(len(test)), annual_drawdown_cols]
            .reset_index(drop=True),
        "rolling_forecast": rolling,
        "rolling_metrics": pd.DataFrame(horizon_metrics),
    }
