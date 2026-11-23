
import numpy as np
import pandas as pd


class Calc:
    def __init__(self, df):
        self.df = df.copy()

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['amount'] = pd.to_numeric(
            self.df['amount'], errors='raise'
        )

        self.df = self.df.sort_values('date').reset_index(drop=True)

    def indicator(self, x):
        x = np.asarray(x, dtype=float)
        return (np.abs(x) <= 1.0) & (x <= 0.0)

    def kernel(self, x):
        return 0.75 * (1.0 - x**2) * self.indicator(x)

    def weight(self, U, X, h):
        U = np.asarray(U, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)

        if h <= 0:
            raise ValueError(
                'Ширина окна h должна быть положительной.'
            )

        if U.size == 0 or X.size == 0:
            raise ValueError(
                'Массивы U и X не должны быть пустыми.'
            )

        XmU = (X[None, :] - U[:, None]) / h

        K = self.kernel(XmU)
        sums = K.sum(axis=1)

        if np.any(~np.isfinite(sums)) or np.any(sums <= 0):
            raise ValueError(
                'Не удалось нормализовать веса: '
                'проверьте h и данные.'
            )

        self.W = K / sums[:, None]
        return self.W

    def NW(self, U, X, Y, h):
        X = np.asarray(X, dtype=float).reshape(-1)
        Y = np.asarray(Y, dtype=float).reshape(-1)

        if X.size != Y.size:
            raise ValueError(
                'X и Y должны иметь одинаковую длину.'
            )

        return self.weight(U, X, h) @ Y

    def LL(self, U, X, Y, h):
        U = np.asarray(U, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)
        Y = np.asarray(Y, dtype=float).reshape(-1)

        if X.size != Y.size:
            raise ValueError(
                'X и Y должны иметь одинаковую длину.'
            )

        W = self.weight(U, X, h)

        alpha = np.empty(U.size, dtype=float)
        beta = np.empty(U.size, dtype=float)

        for i, u in enumerate(U):
            Z1 = (X - u) / h

            Z = np.column_stack((
                np.ones_like(Z1),
                Z1
            ))

            A = Z.T @ (W[i, :, None] * Z)
            B = Z.T @ (W[i] * Y)

            coef = np.linalg.solve(A, B)

            alpha[i], beta[i] = coef

        return alpha, beta

    def low(self, report_dt, h):
        report_dt = pd.Timestamp(report_dt)

        df = self.df.loc[
            self.df['date'].between(
                report_dt - pd.Timedelta(days=h),
                report_dt
            )
        ].copy()

        current = df.loc[
            df['date'] == report_dt,
            'amount'
        ]

        if current.empty:
            raise ValueError(
                f'В данных отсутствует отчётная дата '
                f'{report_dt.date()}.'
            )

        X = (
            df['date'] - report_dt
        ).dt.days.to_numpy(dtype=float)

        Y = df['amount'].to_numpy(dtype=float)

        y_hat = self.NW([0.0], X, Y, h)[0]

        return bool(current.iloc[0] < y_hat)

    def bucket_calc_no_dec(
        self,
        report_dt,
        a_low,
        a_high,
        beta,
        h
    ):
        report_dt = pd.Timestamp(report_dt)

        current = self.df.loc[
            self.df['date'] == report_dt,
            'amount'
        ]

        if current.empty:
            raise ValueError(
                f'В данных отсутствует отчётная дата '
                f'{report_dt.date()}.'
            )

        year_mask = self.df['date'].between(
            report_dt - pd.DateOffset(years=1),
            report_dt
        )

        history = self.df.loc[
            year_mask,
            'amount'
        ]

        if history.empty:
            raise ValueError(
                'Нет данных за год до отчётной даты.'
            )

        amount = float(current.iloc[0])

        stable = float(history.min() * beta)

        self.df['stable'] = stable

        variable = amount - stable

        a = np.asarray(
            a_low if self.low(report_dt, h) else a_high,
            dtype=float
        )

        if a.size == 0:
            raise ValueError(
                'Список коэффициентов a не должен быть пустым.'
            )

        if not np.all(np.isfinite(a)):
            raise ValueError(
                'Коэффициенты a содержат NaN или бесконечность.'
            )

        x = np.zeros(a.size, dtype=float)

        x[0] = variable * a[0]

        for i in range(1, a.size):
            x[i] = (
                min(a[i] - np.min(a[:i]), 0.0)
                * variable
            )

        return x
