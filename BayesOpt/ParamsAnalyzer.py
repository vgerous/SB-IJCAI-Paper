import json
import math 
import copy
import random
import numpy as np
import sys
sys.path.append("..")

from nongrad_models.GaussianMixtureModel import GmmModel
from nongrad_models.LinearFitModel import LinearFitModel
from nongrad_models.TSDecModel import TSDecModel
from grad_models import ReluFCNet
from .bo_config import TSDec_bo_config


class ParamsAnalyzer:
    '''
    ParamsAnalyzer for unit-models solver
    '''
    def __init__(self, config, T, input_size, random_seed=1):
        np.random.seed(random_seed)
        random.seed(random_seed+1)

        self.mapping = {
            "TSDecModel": TSDecModel(T=T),
            "LinearFitModel": LinearFitModel(T=T),
            "FCNet": ReluFCNet(input_size=input_size, T=T, pred_size=T),
            "GMM": GmmModel(),
        }

        self.predictor_name = [*config["predictor_config"].keys()][0]
        self.predictor_config = copy.deepcopy(config["predictor_config"][self.predictor_name])
        
        index = 0
        for param_name, param_info in self.predictor_config.items():
            self.predictor_config[param_name]["begin_index"] = index
            if param_info["type"] in ["int", "float", "real"]:
                index += 1
            elif param_info["type"] in ["category", "bool"]:
                index += len(param_info["range"])
            else:
                raise Exception("unsupport parameter type!")

        self.uncertainty_name = [*config["uncertainty_config"].keys()][0]
        self.uncertainty_config = copy.deepcopy(config["uncertainty_config"][self.uncertainty_name])

        for param_name, param_info in self.uncertainty_config.items():
            self.uncertainty_config[param_name]["begin_index"] = index
            if param_info["type"] in ["int", "float", "real"]:
                index += 1
            elif param_info["type"] in ["category", "bool"]:
                index += len(param_info["range"])
            else:
                raise Exception("unsupport parameter type!")  

        self.vector_length = index

    def _sample(self):
        vector = [0] * self.vector_length
        for param_name, param_info in [*self.predictor_config.items()] + [*self.uncertainty_config.items()]:
            if param_info["type"] == "int":
                param_range = param_info["range"]
                vector[param_info["begin_index"]] = np.random.randint(param_range[0], param_range[1])
            elif param_info["type"] in ["category", "bool"]:
                possible_values = param_info["range"]
                sampled_index = np.random.randint(0, len(possible_values))
                vector[param_info["begin_index"]+sampled_index] = 1
            elif param_info["type"] == "float":
                possible_values = param_info["range"]
                vector[param_info["begin_index"]] = random.choice(possible_values)
            elif param_info["type"] == "real":
                param_range = param_info["range"]
                vector[param_info["begin_index"]] = np.random.uniform(param_range[0], param_range[1])
            else:
                raise Exception("unsupport parameter type!")
        return vector

    def sample(self, n):
        sample_result = [self._sample() for _ in range(n)]
        return sample_result
    
    def analyse(self, X):
        predictor_paramset = {}
        for param_name, param_info in self.predictor_config.items():
            if param_info["type"] in ["int", "float", "real"]:
                value = X[param_info["begin_index"]]
                predictor_paramset[param_name] = value
            elif param_info["type"] in ["category", "bool"]:
                for i in range(len(param_info["range"])):
                    if X[param_info["begin_index"] + i] == 1:
                        predictor_paramset[param_name] = param_info["range"][i]
                        break
            else:
                raise Exception("unsupport parameter type!")

        uncertainty_paramset = {}
        for param_name, param_info in self.uncertainty_config.items():
            if param_info["type"] in ["int", "float", "real"]:
                uncertainty_paramset[param_name] = X[param_info["begin_index"]]
            elif param_info["type"] in ["category", "bool"]:
                for i in range(len(param_info["range"])):
                    if X[param_info["begin_index"] + i] == 1:
                        uncertainty_paramset[param_name] = param_info["range"][i]
                        break
            else:
                raise Exception("unsupport parameter type!")  

        predictor = copy.deepcopy(self.mapping[self.predictor_name])
        if len(predictor_paramset)> 0:
            predictor.set_params(**predictor_paramset)
        uncertainty = copy.deepcopy(self.mapping[self.uncertainty_name])
        if len(uncertainty_paramset)> 0:
            uncertainty.set_params(**uncertainty_paramset)
        
        return predictor, uncertainty