from .BayesianOptimization import BayesianOptimization
from .acquisition_function import ei_acquisition
from .ParamsAnalyzer import ParamsAnalyzer

class BO_main:
    def __init__(self, bo_config, T, object_function, input_size=1, max_iter=10):
        self.bo_params = {
            "max_iter": max_iter,
            "init_sample": 1,
            "each_loop_sample": 1000,
            "exp_rate": 0.9,
            "random_seed": 1
        }

        self.obj_func = object_function
        self.pa = ParamsAnalyzer(bo_config, T, input_size)
        self.bo = BayesianOptimization(object_function=self.obj_func, params_analyzer=self.pa, **self.bo_params)


    def run(self):
        best_X, _ = self.bo.run()
        best_model = self.pa.analyse(best_X)

        return best_model