
import pandas as pd

import calc
from corp_ca_calibration import calibration


DEFAULT_TIME = (30, 61, 91, 122, 152, 183, 213, 244, 274, 305, 335, 365)
DEFAULT_BUCKETS = (
    '1M', '2M', '3M', '4M', '5M', '6M',
    '7M', '8M', '9M', '10M', '11M', '1Y'
)


def get_bst_params(segment):
    params = {
        'ul_rub': (20, 15),
        'ul_byn': (30, 20),
        'fl_rub': (148, 14),
        'fl_byn': (30, 20),
    }
    if segment not in params:
        raise ValueError('Некорректное имя сегмента.')
    return params[segment]


def _prepare_dataframe(df):
    if not {'date', 'amount'}.issubset(df.columns):
        raise ValueError(
            "DataFrame должен содержать столбцы 'date' и 'amount'."
        )

    result = df.copy()
    result['date'] = pd.to_datetime(result['date'], errors='raise')
    result['amount'] = pd.to_numeric(result['amount'], errors='raise')

    if result[['date', 'amount']].isna().any().any():
        raise ValueError(
            'Столбцы date и amount не должны содержать пропуски.'
        )

    return result.sort_values('date').reset_index(drop=True)


def calibrate(
    df,
    calibration_date,
    bw,
    percentile=1,
    k_type='Epanechnikov',
    time=DEFAULT_TIME,
    buckets=DEFAULT_BUCKETS,
):
    if len(time) != len(buckets):
        raise ValueError(
            'Количество корзин и горизонтов должно совпадать.'
        )

    calibration_date = pd.Timestamp(calibration_date)

    df1 = _prepare_dataframe(df)

    df1 = df1.loc[
        df1['date'] <= calibration_date
    ].reset_index(drop=True)

    if df1.empty:
        raise ValueError(
            'На дату калибровки нет исторических наблюдений.'
        )

    df1 = df1.reset_index()

    model = calibration(df1)

    beta = model.stable()

    model.preproc(buckets, time)

    at_low, at_high = model.at_calc_NW(
        percentile,
        buckets,
        bw,
        ktype=k_type,
        x_name='index'
    )

    return {
        'model': model,
        'beta': beta,
        'at_low': at_low,
        'at_high': at_high,
    }


def build_model_gap(
    df,
    calibration_date,
    report_dt,
    bw,
    percentile,
    k_type='Epanechnikov',
    time=DEFAULT_TIME,
    buckets=DEFAULT_BUCKETS,
):
    if len(time) != len(buckets):
        raise ValueError(
            'Количество корзин и горизонтов должно совпадать.'
        )

    report_dt = pd.Timestamp(report_dt)

    df1 = _prepare_dataframe(df)

    calib_res = calibrate(
        df1,
        calibration_date,
        bw=bw,
        percentile=percentile,
        k_type=k_type,
        time=time,
        buckets=buckets,
    )

    outflow = calc.Calc(df1).bucket_calc_no_dec(
        report_dt=report_dt,
        a_low=calib_res['at_low'],
        a_high=calib_res['at_high'],
        beta=calib_res['beta'],
        h=bw,
    )

    return pd.DataFrame(
        [outflow],
        index=[report_dt],
        columns=buckets
    )
