import time
import numpy as np
import warnings
from .acquisition_function import ei_acquisition
from .ParamsAnalyzer import ParamsAnalyzer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import datetime

class BayesianOptimization:
    def __init__(self, object_function, params_analyzer, acquisition_function=ei_acquisition, max_iter=300, init_sample=1, each_loop_sample=1000, exp_rate=0.9, random_seed=1):
        self.object_function = object_function
        self.acquisition_function = acquisition_function
        self.max_iter = max_iter
        self.pa = params_analyzer
        self.init_sample = init_sample
        self.each_loop_sample = each_loop_sample
        self.exp_rate = exp_rate
        self.random_seed = random_seed

        np.random.seed(random_seed)
    
    def _get_next_model(self, ac):
        '''
        Adding some randomness
        '''
        sorted_ac = sorted(ac, key=lambda x: x[1][0], reverse=True)[:25]
        if np.random.random() > self.exp_rate:
            sorted_ac = sorted(sorted_ac, key=lambda x: x[1][1], reverse=True)
        return sorted_ac[0][0]


    def run(self):
        '''
        the main bayesian optimization loop
        '''
        
        # init
        self.X = self.pa.sample(self.init_sample)
        if self.max_iter <=0:
            return self.X, None
        init_models = [self.pa.analyse(x) for x in self.X]

        self.y = [self.object_function(prediction_model=m[0], uncertainty_model=m[1]) for m in init_models]
        self.best_x = self.X[np.argmax(self.y)]
        self.best_y = max(self.y)
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), 
                                           alpha=1e-8,
                                           normalize_y=True,
                                           n_restarts_optimizer=5,
                                           random_state=self.random_seed+10086)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(self.X, self.y)

        iter_times = 0
        while iter_times < self.max_iter:
            print(f"Running iteration {iter_times} of {self.max_iter} @ {datetime.datetime.utcnow()}", flush=True)
            sample_X = self.pa.sample(self.each_loop_sample)
            ac = [[i, self.acquisition_function([sample_X[i]], gp=self.gp, y_max=self.best_y, xi=0.1)] for i in range(self.each_loop_sample)]
            next_model_index = self._get_next_model(ac)
            next_model = self.pa.analyse(sample_X[next_model_index])
            new_x = sample_X[next_model_index]
            new_y = self.object_function(prediction_model=next_model[0], uncertainty_model=next_model[1]) 
            if new_y > self.best_y:
                self.best_x = new_x
                self.best_y = new_y

            self.X.append(new_x)
            self.y.append(new_y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.X, self.y)
            iter_times += 1
        return self.best_x, self.best_y
