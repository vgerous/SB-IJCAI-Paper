import numpy as np
from pmdarima.arima import auto_arima
import warnings

warnings.filterwarnings("ignore")

class AutoARIMAModel:
    def __init__(self, T, **autoarima_params):
        self.T = T
        self.autoarima_params = autoarima_params
        self.model_name = "Auto_ARIMA_%s_Model" % str(self.autoarima_params)

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        y = []
        for data in X:
            self._fit(data)
            y.append(self._predict(self.T))
        return np.array(y)

    def _fit(self, data):
        self.data = np.array(data)
        self.model = auto_arima(self.data, error_action="ignore", n_jobs=-1, **self.autoarima_params)
        self.model.fit(self.data)

    def _predict(self, next_n_prediction):
        prediction = self.model.predict(next_n_prediction)
        prediction = [round(p) if p > 0 else 0 for p in prediction]
        return prediction
