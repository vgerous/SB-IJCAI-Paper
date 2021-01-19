import numpy as np
from sklearn.linear_model import LinearRegression


class LinearFitModel:
    def __init__(self, T, latest_n=60):
        self.model_name = "LinearFit_{}_Model".format(latest_n)
        self.T = T
        self.latest_n = latest_n

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        y = []
        for data in X:
            self._fit(data)
            y.append(self._predict(self.T))
        return np.array(y)

    def _fit(self, data):
        x = np.array([[i + 1] for i in range(len(data[-self.latest_n:]))])
        self.model = LinearRegression()
        self.model.fit(x, np.array(data[-self.latest_n:]))

    def _predict(self, next_n_prediction):
        x_next = np.array([[self.latest_n + i + 1] for i in range(next_n_prediction)])
        pred = self.model.predict(x_next)
        pred = [ele if ele > 0 else 0 for ele in pred]
        return pred

    def set_params(self, **params):
        if "latest_n" in params:
            self.latest_n = params["latest_n"]
        self.model_name = "LinearFit_{}_Model".format(self.latest_n)