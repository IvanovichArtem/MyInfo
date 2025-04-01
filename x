import numpy as np
from scipy.optimize import least_squares

class AlphaCalibration:
    def __init__(self, X, valid_matrix, r_s_valid):
        """
        X — матрица миграций, построенная на train
        valid_matrix — матрица, построенная на valid
        r_s_valid — коэффициент дефолтов (скаляр), посчитанный на valid
        """
        assert X.shape == valid_matrix.shape, "X и valid_matrix должны иметь одинаковую размерность."
        self.X = X.values
        self.valid_matrix = valid_matrix.values
        self.r_s_valid = r_s_valid
        self.shape = X.shape
        # Начальное приближение для матрицы alpha: единицы
        self.alpha0 = np.ones(self.shape).flatten()

    def residuals(self, alpha_flat):
        """
        Функция остатков: разница между предсказанной и фактической valid_matrix.
        Предсказанная valid_matrix = X + alpha * (r_s_valid - 1)
        """
        alpha = alpha_flat.reshape(self.shape)
        pred = self.X + alpha * (self.r_s_valid - 1)
        return (pred - self.valid_matrix).flatten()

    def calibrate(self):
        """
        Калибрует матрицу alpha с использованием метода наименьших квадратов.
        Ограничения: alpha >= 0
        """
        result = least_squares(self.residuals, self.alpha0, bounds=(-1, 1))
        self.alpha_opt = result.x.reshape(self.shape)
        return self.alpha_opt
