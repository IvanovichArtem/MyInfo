
from math import ceil

import numpy as np
import pandas as pd


class calibration:
    def __init__(self, df):
        self.df = df.copy()
        self.beta = None

    def stable(self):
        st = self.df[['date', 'amount']].copy()

        st['min_past'] = (
            st['amount']
            .rolling(365, min_periods=100)
            .min()
        )

        st['min_future'] = (
            st['amount']
            .iloc[::-1]
            .rolling(365, min_periods=100)
            .min()
            .iloc[::-1]
        )

        st['beta'] = (
            st['min_future'] / st['min_past']
        )

        valid_beta = (
            st['beta']
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if valid_beta.empty:
            raise ValueError(
                'Невозможно вычислить beta: '
                'недостаточно корректных данных.'
            )

        self.beta = float(
            min(
                valid_beta.min() // 0.0001 / 10000,
                1.0
            )
        )

        return self.beta

    def indicator(self, x):
        x = np.asarray(x, dtype=float)

        return (x >= -1.0) & (x <= 0.0)

    def kernel(self, x, ktype='Epanechnikov'):
        x = np.asarray(x, dtype=float)

        if ktype == 'Truncated':
            return 0.5 * self.indicator(x)

        if ktype == 'Epanechnikov':
            return (
                0.75 * (1.0 - x**2)
                * self.indicator(x)
            )

        if ktype == 'Gaussian':
            return (
                np.exp(-0.5 * x**2)
                / np.sqrt(2.0 * np.pi)
                * (x <= 0.0)
            )

        raise ValueError(
            f'Неизвестное ядро: {ktype!r}'
        )

    def weight(
        self,
        U,
        X,
        h,
        ktype='Epanechnikov'
    ):
        U = np.asarray(U, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)

        if h <= 0:
            raise ValueError(
                'Ширина окна h должна быть положительной.'
            )

        if X.size == 0 or U.size == 0:
            raise ValueError(
                'Массивы наблюдений и точек оценки '
                'не должны быть пустыми.'
            )

        XmU = (
            X[None, :] - U[:, None]
        ) / h

        K = self.kernel(
            XmU,
            ktype=ktype
        )

        sums = K.sum(axis=1)

        if (
            np.any(~np.isfinite(sums))
            or np.any(sums <= 0)
        ):
            raise ValueError(
                'Невозможно вычислить веса: '
                'для части точек нет наблюдений '
                'в пределах ядра. '
                'Проверьте bw, даты и данные.'
            )

        return K / sums[:, None]

    def NW(
        self,
        U,
        X,
        Y,
        h,
        ktype='Epanechnikov'
    ):
        Y = np.asarray(Y, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float).reshape(-1)

        if X.size != Y.size:
            raise ValueError(
                'X и Y должны иметь одинаковую длину.'
            )

        return (
            self.weight(
                U, X, h,
                ktype=ktype
            ) @ Y
        )

    def preproc(self, buckets, time):
        if self.beta is None:
            raise ValueError(
                'Сначала вызовите stable().'
            )

        if len(buckets) != len(time):
            raise ValueError(
                'Количество корзин и горизонтов '
                'должно совпадать.'
            )

        self.df['stable'] = (
            self.df['amount']
            .rolling(365)
            .min()
            * self.beta
        )

        self.df['variable'] = (
            self.df['amount']
            - self.df['stable']
        )

        for bucket, t in zip(buckets, time):
            t = int(t)

            if t < 1:
                raise ValueError(
                    'Горизонты должны быть положительными.'
                )

            rolling_min = (
                self.df['amount']
                .rolling(
                    window=t + 1,
                    min_periods=ceil(t / 2)
                )
                .min()
            )

            self.df[str(t)] = rolling_min

            with np.errstate(
                divide='ignore',
                invalid='ignore'
            ):
                ratio = (
                    rolling_min.shift(-t)
                    - self.df['amount']
                ) / self.df['variable']

            self.df[bucket] = ratio.replace(
                [np.inf, -np.inf],
                np.nan
            )

    def at_calc_NW(
        self,
        percentile,
        buckets,
        bw,
        ktype='Epanechnikov',
        x_name='index'
    ):
        if not 0 <= percentile <= 100:
            raise ValueError(
                'percentile должен находиться '
                'в диапазоне от 0 до 100.'
            )

        if (
            not isinstance(bw, (int, np.integer))
            or bw < 1
            or bw >= len(self.df)
        ):
            raise ValueError(
                'bw должен быть целым числом '
                'от 1 до len(df) - 1.'
            )

        if not buckets:
            raise ValueError(
                'Список buckets не должен быть пустым.'
            )

        if x_name == 'index':
            X = self.df.index.to_numpy(
                dtype=float
            )

        elif x_name in self.df.columns:
            X = self.df[x_name].to_numpy(
                dtype=float
            )

        else:
            raise ValueError(
                f'Неизвестная переменная '
                f'x_name: {x_name!r}'
            )

        U = X[bw:]

        Y = self.df['amount'].to_numpy(
            dtype=float
        )

        self.X_pred = self.NW(
            U,
            X,
            Y,
            h=bw,
            ktype=ktype
        )

        self.df['X_pred'] = np.nan

        self.df.loc[
            self.df.index[bw:],
            'X_pred'
        ] = self.X_pred

        low = self.df.loc[
            self.df['amount'] < self.df['X_pred']
        ]

        high = self.df.loc[
            self.df['amount'] >= self.df['X_pred']
        ]

        def coefficients(sample, group_name):
            result = []

            for bucket in buckets[:-1]:
                values = (
                    sample[bucket]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan
                    )
                    .dropna()
                    .to_numpy(dtype=float)
                )

                if values.size == 0:
                    raise ValueError(
                        f'Нет наблюдений для расчёта '
                        f'{group_name}, '
                        f'корзина {bucket!r}. '
                        f'Проверьте динамику остатков '
                        f'и период данных.'
                    )

                value = np.percentile(
                    values,
                    percentile,
                    method='lower'
                )

                result.append(
                    float(
                        value // 0.0001 / 10000
                    )
                )

            result.append(-1.0)

            return result

        print('Всего наблюдений:', len(self.df))
        print('Low:', len(low))
        print('High:', len(high))

        print('Low 1M:', low['1M'].notna().sum())
        print('High 1M:', high['1M'].notna().sum())

        a_low = coefficients(low, 'a_low')
        a_high = coefficients(high, 'a_high')

        return a_low, a_high
