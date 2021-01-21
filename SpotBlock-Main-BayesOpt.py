#!/usr/bin/env python
# coding: utf-8

import warnings
import sys
import numpy as np
from scipy.stats import norm
import time
import torch

# local libraries & helper functions
from utils import load_data, scale_and_downsample, load_GroundTruth_utilities, parse_arguments
from qpth_utils import collect_ML_data, eval_perf_batch, get_parallel_executor
from nongrad_models.GaussianMixtureModel import GmmModel
from BayesOpt.BO_main import BO_main
from BayesOpt.bo_config import TSDec_bo_config, FCNet_bo_config, LinearFit_bo_config, LstmNet_bo_config


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # retrieve params
    args = parse_arguments()
    # retrieve params
    branch = args.branch
    d_factor = args.d_factor
    model_name = args.model_name
    if_optnet = args.if_optnet
    optnet_vio_regularity = args.optnet_vio_regularity
    optnet_iterations = args.optnet_iterations
    p = args.p # violation threshold

    on_gpu = torch.cuda.is_available() if args.use_gpu is None else args.use_gpu
    nn_device = torch.device('cuda') if on_gpu else torch.device('cpu')

    file_name = 'BayesOpt_' + '_'.join([
        branch,
        str(d_factor),
        model_name,
        str(if_optnet),
        str(args.p),
        str(args.bo_iter),
        args.opt,
        *(('SkipUncertainity', ) if args.skip_uncertainity else ())
    ])
    # generate filename to write results to
    if if_optnet == 1:
        file_name = file_name+"_"+str(optnet_vio_regularity)+"_"+str(optnet_iterations)

    # load capacity and requests data
    print("loading data --")
    capacities, requests = load_data('data/', branch)
    print("loading complete.")
    print("totally", len(capacities), "hours of capacity data.")
    print("total number of requests:", len(requests))

    if branch == 'Default':
        request_thres = 1e4 * d_factor
    if branch == 'Azure2019':
        request_thres = 0
    if branch == 'Azure2017':
        request_thres = 0

    # scale and downsample
    request_downsampling_factor = d_factor  # ratio of requests down-sampling
    capacity_scale_factor = d_factor     # factor for scaling down the capacities


    print("scaling data ---")
    print("capacity_scale_factor:", capacity_scale_factor)
    print("request_downsampling_factor:", request_downsampling_factor)
    scaled_capacities, DownSampledRequests = scale_and_downsample(
        capacities, requests, capacity_scale_factor, request_downsampling_factor)

    # collect relevant ML data
    history_lookback = 120          # how many history timesteps used for prediction
    T = 24                          # how long a timewindow of an instance is
    """
    data_X1: history capacities
    req_params: current requests data
    data_y: ground truth current capacities
    """
    print("transforming data to ML form --")
    data_X1, req_params, data_y = collect_ML_data(
        history_lookback, T, request_thres, scaled_capacities, DownSampledRequests)
    print("data_X1.shape:", data_X1.shape)
    print("data_y.shape:", data_y.shape)

    # split training & testing data
    train_ratio = 0.75
    num_train_samples = round(train_ratio*len(data_X1))
    X1_train = data_X1[:num_train_samples]
    X1_test = data_X1[num_train_samples:]
    req_params_train = req_params[:num_train_samples]
    req_params_test = req_params[num_train_samples:]
    y_train = data_y[:num_train_samples]
    y_test = data_y[num_train_samples:]

    # ------------------------- data collection & transformation complete ------------------------

    # load pre-calculated ground truth utilities
    try:
        utilities_True = load_GroundTruth_utilities('exp_results/GroundTruth_results/', branch, d_factor)
    except:
        print("[ERROR]: no pre-cached ground truth results.")
        assert False

    # ----------------------------- begin model training & evaluation ----------------------------
    def bayesOpt_objFunc(prediction_model, uncertainty_model, lambda_p=1e4):
        # calculate the models' performance on the **training set** and obtain a final score
        data_scaler = np.max(X1_train)/10
        prediction_model.fit(X1_train / data_scaler, y_train / data_scaler)
        preds_train = prediction_model.predict(X1_train / data_scaler) * data_scaler

        E = y_train - preds_train
        # calculate the fitted um attributes for each timestamp
        delta = []  # capacity cutdowns; array of length T.
        for t in range(E.shape[1]):
            if uncertainty_model is None:
                delta.append(0)
                continue
            errors_t = E[:, t]
            uncertainty_model.fit(errors_t)
            epsilon = uncertainty_model.get_distribution()
            mu = epsilon.exp  # error expectation
            stdev = epsilon.std  # error standard deviation
            delta.append(-norm.ppf(p) * stdev - mu)
        delta = np.array(delta)

        print("Delta:")
        print(delta)
        print("y_train:")
        print(y_train)
        print("preds_train:")
        print(preds_train)
        # evaluate the RobustOpt solution's performance on the training set
        utilities, violating_usage, viorate, _, run_batch_total_duration = eval_perf_batch(
            y_train, preds_train-delta, req_params_train, T, args.opt, time_lim=30, display=False, parallel=args.parallel, parallel_proc=args.proc)
        aver_utilities = np.mean(utilities)
        aver_viorate = np.mean(viorate)

        # note: the goal is to maximize this output
        return aver_utilities - lambda_p*max(aver_viorate-p, 0)

    # some verifications: choose a prediction model and a uncertainty model
    print("using model", model_name)
    if model_name == "FCNet":
        from grad_models import ReluFCNet
        net = ReluFCNet(input_size=X1_train.shape[1], hid1=100, hid2=50, pred_size=T, T=T,
                        SGD_params=[1e-4, 0.9], nn_device=nn_device)
        BayesOpt_selector = BO_main(FCNet_bo_config, T, object_function=bayesOpt_objFunc, input_size=X1_train.shape[1],
                                    nn_device=nn_device, max_iter=args.bo_iter, skip_uncertainity = args.skip_uncertainity)
    elif model_name == "LstmNet":
        from grad_models import LstmNet
        net = LstmNet(input_feature_dim=1, output_size=T, T=T, hid_size=100, step_size=1e-3,
                      nn_device=nn_device)
        BayesOpt_selector = BO_main(LstmNet_bo_config, T, bayesOpt_objFunc, nn_device=nn_device, max_iter=args.bo_iter, skip_uncertainity = args.skip_uncertainity)
    elif model_name == "LinearFit":
        from nongrad_models.LinearFitModel import LinearFitModel
        net = LinearFitModel(T=T, latest_n=60)
        BayesOpt_selector = BO_main(LinearFit_bo_config, T, bayesOpt_objFunc, nn_device=nn_device, max_iter=args.bo_iter, skip_uncertainity = args.skip_uncertainity)
    elif model_name == "TSDec":
        from nongrad_models.TSDecModel import TSDecModel
        net = TSDecModel(T=T, decompose_length=360, trend_length=12, residual_length=360,
                         time_series_frequency=12)
        BayesOpt_selector = BO_main(TSDec_bo_config, T, bayesOpt_objFunc, nn_device=nn_device, max_iter=args.bo_iter, skip_uncertainity = args.skip_uncertainity)
    else:
        assert False
    um = GmmModel(tol=1e-10, max_components=100) if not args.skip_uncertainity else None

    # find BO-optimized models
    tick = time.time()
    net_final, um_final = BayesOpt_selector.run()
    tock = time.time()
    timing_train_time = tock - tick
    def evaluate_BO_testset(net_final, um_final):
        # evaluate the model on the test set
        data_scaler = np.max(X1_train)/10
        net_final.fit(X1_train / data_scaler, y_train / data_scaler)
        preds_train = net_final.predict(X1_train / data_scaler) * data_scaler
        preds_test = net_final.predict(X1_test / data_scaler) * data_scaler
        E = y_train - preds_train       # the errors on the training set
        delta = []  # capacity cutdowns; array of length T.
        for t in range(E.shape[1]):
            if um_final is None:
                delta.append(0)
                continue
            errors_t = E[:, t]
            um_final.fit(errors_t)
            epsilon = um_final.get_distribution()
            mu = epsilon.exp  # error expectation
            stdev = epsilon.std  # error standard deviation
            delta.append(-norm.ppf(p) * stdev - mu)
        delta = np.array(delta)
        # evaluate the performance on the **TEST** set
        utilities, violating_usage, viorate, _, run_batch_total_duration = eval_perf_batch(
            y_test, preds_test-delta, req_params_test, T, args.opt, time_lim=30, display=False, parallel=args.parallel, parallel_proc=args.proc)
        utilities_ratio = np.array(utilities) / utilities_True
        aver_utilities = np.mean(utilities)
        aver_util_ratio = np.mean(utilities_ratio)
        aver_viorate = np.mean(viorate)
        
        std_utilities = np.std(utilities)
        std_util_ratio = np.std(utilities_ratio)
        std_viorate = np.std(viorate)
        return aver_utilities, aver_util_ratio, aver_viorate, std_utilities, std_util_ratio, std_viorate


    tick = time.time()
    train_eval_score_before = bayesOpt_objFunc(net, um)
    tock = time.time()
    timing_eval_time_before = tock - tick

    
    tick = time.time()
    train_eval_score_after = bayesOpt_objFunc(net_final, um_final)
    tock = time.time()
    timing_eval_time_after = tock - tick

    print("[TRAIN SET] evaluation score before BO: ", train_eval_score_before,
          " and after BO: ", train_eval_score_after)


    tick = time.time()
    evaluation_result_before = evaluate_BO_testset(net, um)
    print("[TEST SET] aver utilities and aver viorate before BO:")
    print(evaluation_result_before)
    tock = time.time()
    timing_test_time_before = tock - tick

    tick = time.time()
    evaluation_result_after = evaluate_BO_testset(net_final, um_final)
    print("[TEST SET] aver utilities and aver viorate after BO:")
    print(evaluation_result_after)
    tock = time.time()
    timing_test_time_after = tock - tick

    with open('exp_results/'+file_name+'.txt', 'w', encoding='utf-8') as f:
        f.write(f"{file_name}\n")
        f.write(f"[Train set]:\n")
        f.write(f" evaluation score before BO: {train_eval_score_before}\n")
        f.write(f" evaluation score after BO: {train_eval_score_after}\n")
        f.write(f"[Test set]:\n")
        f.write(f" aver utilities, util ratio and aver viorate before BO: {evaluation_result_before[0]}, {evaluation_result_before[1]}, {evaluation_result_before[2]}\n")
        f.write(f" aver utilities, util ratio and aver viorate after BO: {evaluation_result_after[0]}, {evaluation_result_after[1]}, {evaluation_result_after[2]}\n")

    with open('exp_results/'+file_name+'.csv', 'w', encoding='utf-8') as f:
        f.write(",".join([
            'branch',
            'd_factor',
            'model_name',
            'if_optnet',
            'args_p',
            'bo_iter',
            'opt',
            'train_time',
            'eval_time_before',
            'eval_time_after',
            'test_time_before',
            'test_time_after',
            'train_eval_score_before',
            'train_eval_score_after',
            'before_aver_utilities',
            'before_aver_util_ratio',
            'before_aver_viorate',
            'before_std_utilities',
            'before_std_util_ratio',
            'before_std_viorate',
            'after_aver_utilities',
            'after_aver_util_ratio',
            'after_aver_viorate',
            'after_std_utilities',
            'after_std_util_ratio',
            'after_std_viorate',
        ]) + "\n")
        f.write(",".join([
            branch,  # branch
            str(d_factor), # d_factor
            model_name, # model_name
            str(if_optnet), # if_optnet
            str(args.p), # args_p
            str(args.bo_iter), # bo_iter
            args.opt, #Opt method
            str(timing_train_time), # train time
            str(timing_eval_time_before), # eval time before
            str(timing_eval_time_after), # eval time after
            str(timing_test_time_before), # test time before
            str(timing_test_time_after), # test time after
            str(train_eval_score_before), # train_eval_score_before
            str(train_eval_score_after), # train_eval_score_after
            str(evaluation_result_before[0]), # before_aver_utilities 
            str(evaluation_result_before[1]), # before_aver_util_ratio
            str(evaluation_result_before[2]), # before_aver_viorate
            str(evaluation_result_before[3]), # before_std_utilities
            str(evaluation_result_before[4]), # before_std_util_ratio
            str(evaluation_result_before[5]), # before_std_viorate
            str(evaluation_result_after[0]), # after_aver_utilities 
            str(evaluation_result_after[1]), # after_aver_util_ratio
            str(evaluation_result_after[2]), # after_aver_viorate
            str(evaluation_result_after[3]), # after_std_utilities
            str(evaluation_result_after[4]), # after_std_util_ratio
            str(evaluation_result_after[5]), # after_std_viorate
        ]) + "\n")

    if args.parallel:
        executor = get_parallel_executor(args.proc)
        executor.shutdown()
