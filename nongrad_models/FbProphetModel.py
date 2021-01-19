import numpy as np
import pandas as pd
from .utils import daterange, SuppressStdoutStderr
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))

def round_non_negative_int_func(arr):
    return [p if p > 0 else 0 for p in arr]

class FbProphetModel:

    def __init__(self, T, add_std_factor = 0, changepoint_prior_scale=0.3, start_date=datetime(2017, 6, 1, 00, 00), silence=True):
        self.T = T
        self.model_name = "FbProphet_%s_%s_Model" % (str(add_std_factor), str(changepoint_prior_scale))
        self.changepoint_prior_scale = changepoint_prior_scale
        self.add_std_factor = add_std_factor
        self.start_date = start_date
        self.silence = silence
        self.model = None

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        y = []
        for data in X:
            self._fit(data)
            y.append(self._predict(self.T, past_n_validation=50))
        return np.array(y)

    def _fit(self, data):
        from fbprophet import Prophet
        ts = [i for i in daterange(self.start_date, self.start_date+timedelta(hours=len(data)))]
        _dic = {'ds': ts, 'y': data}
        _df = pd.DataFrame(data=_dic)
        if self.silence:
            try:
                with SuppressStdoutStderr():
                    self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale).fit(_df)
            except:
                print("something is wrong with silence mode, disabled.")
                self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale).fit(_df)
        else:
            self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale).fit(_df)

    def _predict(self, next_n_prediction, past_n_validation=1):
        _past_and_future = self.model.make_future_dataframe(periods=next_n_prediction, freq='H')
        _forecast = self.model.predict(_past_and_future)
        _recent_future_pred = _forecast.tail(next_n_prediction + past_n_validation)

        _recent_future_pred_array = _recent_future_pred["yhat"] + self.add_std_factor * (_recent_future_pred["yhat_upper"] - _recent_future_pred["yhat"])
        _recent_future_pred_array = [round(ele) if ele > 0 else 0 for ele in _recent_future_pred_array]

        _validate_pred = _recent_future_pred_array[0:past_n_validation]
        _future_pred = _recent_future_pred_array[-next_n_prediction:]
        return _future_pred

    def set_params(self, **params):
        if "add_std_factor" in params:
            self.add_std_factor = params["add_std_factor"]
        if "changepoint_prior_scale" in params:
            self.changepoint_prior_scale = params["changepoint_prior_scale"]
        if "start_date" in params:
            self.start_date = params["start_date"]
        if "silence" in params:
            self.silence = params["silence"]

        self.model_name = "FbProphet_%s_%s_Model" % (str(self.add_std_factor), str(self.changepoint_prior_scale))
