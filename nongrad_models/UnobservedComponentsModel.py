import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings

class UnobservedComponentsModel:
    def __init__(self, T, **kwargs):
        self.T = T
        self.model_config = kwargs
        self._update_model_name()

    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        y = []
        for data in X:
            self._fit(data)
            y.append(self._predict(self.T))
        return np.array(y)

    def _update_model_name(self):
        self.model_name = "UCM"
        if "level" in self.model_config and self.model_config["level"]:
            self.model_name += "_%s_LEVEL" % self.model_config["level"].replace(" ","_")
        else:
            self.model_name += "_no_LEVEL"
        self.model_name += "_has"
        for component in ["trend", "cycle", "irregular", "stochastic_level", "stochastic_trend", "stochastic_seasonal", "stochastic_cycle", "damped_cycle"]:
            if component in self.model_config and self.model_config[component]:
                self.model_name += "_%s" % component
        for component in ["seasonal", "autoregressive" ]:
            if component in self.model_config and self.model_config[component] is not None:
                self.model_name += "_%d-%s" % (self.model_config[component], component)
        for component in ["freq_seasonal", "exog", "stochastic_freq_seasonal"]:
            if component in self.model_config and self.model_config[component] is not None:
                self.model_name += "_%s" % component
        self.model_name += "_COMPONENTS"
        if "cycle_period_bounds" in self.model_config and self.model_config["cycle_period_bounds"] is not None:
            self.model_name += "_cycle_period_bounds_are_%s" %str(self.model_config["cycle_period_bounds"])
        if "use_exact_diffuse" in self.model_config and self.model_config["use_exact_diffuse"]:
            self.model_name += "_use_exact_diffuse"
        self.model_name += "_Model"

    def _fit(self, data):
        model = UnobservedComponents(endog=data, **self.model_config, irregular = True, level = True, trend = True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trained_model = model.fit(disp=0)

    def _predict(self, next_n_prediction):
        prediction = self.trained_model.forecast(steps=next_n_prediction)
        return [max(0, ele) for ele in prediction]

    def set_params(self, **kwargs):
        self.model_config = kwargs
        self._update_model_name()