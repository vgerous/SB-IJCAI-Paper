TSDec_bo_config = {
    "predictor_config":{
        "TSDecModel": {
            "trend_length": {
                "type": "int",
                "range": [6, 18]
            },
            "time_series_frequency": {
                "type": "int",
                "range": [6, 18]
            },
            "residual_length": {
                "type": "int",
                "range": [300, 360]
            },
            "decompose_length": {
                "type": "int",
                "range": [300, 360]
            }
        }
    },
    "uncertainty_config": {
        "GMM": {
            "max_components":{
                "type": "int",
                "range": [50, 100]
            }
        }
    }
}

FCNet_bo_config = {
    "predictor_config":{
        "FCNet": {
            "hid1": {
                "type": "int",
                "range": [60, 100]
            },
            "hid2": {
                "type": "int",
                "range": [20, 60]
            },
            "step_size": {
                "type": "real",
                "range": [0.0001, 0.001]
            },
            "momentum": {
                "type": "real",
                "range": [0.8, 0.9]
            }
        }
    },
    "uncertainty_config": {
        "GMM": {
            "max_components":{
                "type": "int",
                "range": [50, 100]
            }
        }
    }
}

LinearFit_bo_config = {
    "predictor_config":{
        "LinearFitModel": {
            "latest_n": {
                "type": "int",
                "range": [40, 80]
            }
        }
    },
    "uncertainty_config": {
        "GMM": {
            "max_components":{
                "type": "int",
                "range": [50, 100]
            }
        }
    }
}

LstmNet_bo_config = {
    "predictor_config":{
        "LstmNet": {
            "hid_size": {
                "type": "int",
                "range": [50, 100]
            },
            "step_size": {
                "type": "real",
                "range": [0.0001, 0.001]
            }
        }
    },
    "uncertainty_config": {
        "GMM": {
            "max_components":{
                "type": "int",
                "range": [50, 100]
            }
        }
    }
}