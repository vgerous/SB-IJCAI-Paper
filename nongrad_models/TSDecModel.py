import time
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.mixture import GaussianMixture
from .LinearFitModel import LinearFitModel
from .UnobservedComponentsModel import UnobservedComponentsModel

from scipy.stats import norm
import sys

class TSDecModel:
    def __init__(self, T, decompose_length=360, trend_length=12, residual_length=360,
                 time_series_frequency=12):
        self.T = T
        self.decompose_length = decompose_length
        self.trend_length = trend_length
        self.residual_length = residual_length
        self.time_series_frequency = time_series_frequency
        self.model_name = "TSDec_{}_{}_Model".format( trend_length, time_series_frequency)

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        y = []
        for data in X:
            self._fit(data)
            y.append(self._predict(data, self.T))
        return np.array(y)

    def _fit(self, data):
        self.data = np.array(data)
        self.decomposed_series = seasonal_decompose(x=self.data[-self.decompose_length: ], model='additive', filt=None, 
                                               freq=self.time_series_frequency, two_sided=False, extrapolate_trend=0)

        self.trend_predictor = LinearFitModel(latest_n=self.trend_length, T=self.T)

        self.trend_predictor._fit(self.data[-self.trend_length: ])     
        self.seasonal = self.decomposed_series.seasonal[self.time_series_frequency: ]
        self.irregular = self.decomposed_series.resid[self.time_series_frequency: ]


    def get_resid(self):
        return self.irregular
        

    def _predict(self, data, next_n_prediction):
        prediction = []
        for i in range(next_n_prediction):
            cur_trend = (sum(self.data[-(self.time_series_frequency - 1):]) + self.trend_predictor._predict(next_n_prediction)[i]) / self.time_series_frequency
            cur_seasonal = self.seasonal[len(self.data + i) % self.time_series_frequency]
            cur_irregular = self.irregular.mean()
            
            prediction.append(cur_trend + cur_seasonal + cur_irregular)

        return prediction

    
    def set_params(self, **params):
        if "trend_length" in params:
            self.trend_length = params["trend_length"]
        if "time_series_frequency" in params:
            self.time_series_frequency = params["time_series_frequency"]
        self.model_name = "TSDec_{}_{}_Model".format(self.trend_length, self.time_series_frequency)