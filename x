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
==============================================================================
class AlphaCalibration:
    def __init__(self, matrices, r_s_valid):
        """
        matrices — список кортежей [(X, valid_matrix), ...], где
                   X — матрица миграций, построенная на train
                   valid_matrix — матрица, построенная на valid
        r_s_valid — коэффициент дефолтов (скаляр), посчитанный на valid
        """
        # Проверяем, что все матрицы имеют одинаковую форму
        shapes = [X.shape for X, valid_matrix in matrices]
        assert len(set(shapes)) == 1, "Все матрицы должны иметь одинаковую размерность."
        
        self.matrices = [(X.values, valid_matrix.values) for X, valid_matrix in matrices]
        self.r_s_valid = r_s_valid
        self.shape = shapes[0]
        # Начальное приближение для матрицы alpha: единицы
        self.alpha0 = np.ones(self.shape).flatten()

    def residuals(self, alpha_flat):
        """
        Функция остатков: разница между предсказанной и фактической valid_matrix
        для всех пар матриц.
        Предсказанная valid_matrix = X + alpha * (r_s_valid - 1)
        """
        alpha = alpha_flat.reshape(self.shape)
        residuals = []
        for X, valid_matrix in self.matrices:
            pred = X + alpha * (self.r_s_valid - 1)
            residuals.append((pred - valid_matrix).flatten())
        return np.concatenate(residuals)

    def calibrate(self):
        """
        Калибрует единую матрицу alpha с использованием метода наименьших квадратов.
        Ограничения: alpha >= 0
        """
        result = least_squares(self.residuals, self.alpha0, bounds=(0, np.inf))
        self.alpha_opt = result.x.reshape(self.shape)
        return self.alpha_opt
