import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def train_catboost(df, max_months=5):
    """Обучает модель на каждый горизонт и возвращает модели и метрики."""
    data = df.copy()
    data["REP_DT"] = pd.to_datetime(data["REP_DT"]).dt.normalize()
    data = data.dropna(subset=["REP_DT"]).sort_values("REP_DT")
    data["OST_VAL"] = pd.to_numeric(data["OST_VAL"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    series_cols = ["CURENCY"] if "CURENCY" in data else []
    feature_cols = data.columns.drop(["REP_DT", "OST_VAL"]).tolist()
    cat_cols = data[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        data[col] = data[col].astype("string").fillna("__missing__")

    dates = data["REP_DT"].drop_duplicates().reset_index(drop=True)
    if len(dates) < 3:
        raise ValueError("Для train, valid и test нужны хотя бы три разные даты.")
    valid_start = dates.iloc[int(0.6 * len(dates))]
    test_start = dates.iloc[int(0.8 * len(dates))]

    target_keys = [*series_cols, "target_dt"]
    balances = data[[*series_cols, "REP_DT", "OST_VAL"]].rename(
        columns={"REP_DT": "target_dt", "OST_VAL": "target"}
    )
    origins = data.drop(columns="OST_VAL")
    models, metrics = {}, []

    for months in range(1, max_months + 1):
        horizon = f"{months}m"
        sample = origins.assign(target_dt=origins["REP_DT"] + pd.DateOffset(months=months))
        sample = sample.merge(balances, on=target_keys, validate="many_to_one").dropna(subset=["target"])

        # Будущие остатки для train и valid должны быть известны к началу следующей выборки.
        train = sample[sample["target_dt"] < valid_start]
        valid = sample[(sample["REP_DT"] >= valid_start) & (sample["target_dt"] < test_start)]
        test = sample[sample["REP_DT"] >= test_start]
        if train.empty or valid.empty or test.empty:
            print(f"{horizon} пропущен — мало данных (train={len(train)}, valid={len(valid)}, test={len(test)}).")
            continue

        model = CatBoostRegressor(
            iterations=500, loss_function="MAE", random_seed=42,
            verbose=False, allow_writing_files=False,
        )
        model.fit(
            train[feature_cols], train["target"], cat_features=cat_cols,
            eval_set=(valid[feature_cols], valid["target"]), early_stopping_rounds=50,
        )
        models[horizon] = model

        for split, rows in [("train", train), ("valid", valid), ("test", test)]:
            errors = rows[[*series_cols, "target"]].copy()
            errors["abs_error"] = (errors["target"] - model.predict(rows[feature_cols])).abs()
            errors["ape"] = 100 * errors["abs_error"] / errors["target"].abs().replace(0, np.nan)
            scores = errors.assign(horizon=horizon, split=split).groupby(
                ["horizon", "split", *series_cols], as_index=False
            ).agg(MAE=("abs_error", "mean"), MAPE_pct=("ape", "mean"), n=("target", "size"))
            metrics.append(scores)

    if not metrics:
        raise ValueError("Для заданных горизонтов не получилось собрать train, valid и test.")
    return models, pd.concat(metrics, ignore_index=True)
