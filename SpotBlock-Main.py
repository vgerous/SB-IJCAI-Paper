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


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # retrieve params
    args = parse_arguments()
    # retrieve params
    branch = args.branch
    d_factor = args.d_factor
    model_name = args.model_name
    optnet_vio_regularity = args.optnet_vio_regularity
    optnet_iterations = args.optnet_iterations

    on_gpu = torch.cuda.is_available() if args.use_gpu is None else args.use_gpu 
    nn_device = torch.device('cuda') if on_gpu else torch.device('cpu')

    data_name_str = 'None'
    model_name_str = model_name if not args.if_optnet else f'{model_name}-Opt-{str(optnet_vio_regularity)}-{str(optnet_iterations)}'

    file_name = 'Baseline_' + '_'.join([
        branch,
        data_name_str,
        str(d_factor),
        model_name_str,
        str(args.if_optnet),
        str(args.opt_time),
    ])

    all_lists = []  # used for writing results to a .csv file
    all_text = ""   # used for writing run times to .txt file
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
    capacity_scale_factor = d_factor        # factor for scaling down the capacities

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

    # scale the data s.t. the maximum value is 10
    max_X1 = np.max(data_X1)
    data_scaler = max_X1/10

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
    print("using model", model_name)
    if model_name == "FCNet":
        from grad_models import ReluFCNet
        net = ReluFCNet(input_size=X1_train.shape[1], hid1=100, hid2=50, pred_size=T, T=T,
                        SGD_params=[1e-4, 0.9], nn_device=nn_device)
    elif model_name == "LstmNet":
        from grad_models import LstmNet
        net = LstmNet(input_feature_dim=1, output_size=T, T=T, hid_size=100, step_size=1e-3,
                      nn_device=nn_device)
    elif model_name == "LinearFit":
        from nongrad_models.LinearFitModel import LinearFitModel
        net = LinearFitModel(T=T, latest_n=60)
    elif model_name == "TSDec":
        from nongrad_models.TSDecModel import TSDecModel
        net = TSDecModel(T=T, decompose_length=360, trend_length=12, residual_length=360,
                         time_series_frequency=12)
    elif model_name == "AutoARIMA":
        from nongrad_models.AutoARIMAModel import AutoARIMAModel
        net = AutoARIMAModel(T=T)
    elif model_name == "FbProphet":
        from nongrad_models.FbProphetModel import FbProphetModel
        net = FbProphetModel(T=T)
    elif model_name == "UCM":
        from nongrad_models.UnobservedComponentsModel import UnobservedComponentsModel
        net = UnobservedComponentsModel(T=T)
    else:
        assert False

    # train and predict
    print("begin training")
    tick = time.time()
    net.fit(X1_train/data_scaler, y_train/data_scaler)
    tock = time.time()
    timing_train_time = tock - tick
    print("finished training")
    # training time
    all_text += "run time for model fitting on training set: "
    all_text += str(timing_train_time)
    all_text += "\n"


    print("begin prediction")
    preds_train = net.predict(X1_train/data_scaler) * data_scaler
    tick = time.time()
    preds_test = net.predict(X1_test/data_scaler) * data_scaler
    tock = time.time()
    timing_pred_time = tock - tick
    print("finished prediction")
    all_text += "run time for model predicting on test set: "
    all_text += str(timing_pred_time)
    all_text += "\n"
    if not (preds_train > 0).all():
        print("[WARNING]: negative capacity predictions on training set.")
        preds_train[preds_train < 0] = 0.01
    if not (preds_test > 0).all():
        print("[WARNING]: negative capacity predictions on training set.")
        preds_test[preds_test < 0] = 0.01

    # --------evaluate the performance (objval & violation) on the test set and save to files-------
    if not args.if_optnet == 1:
        # No training with opt for two stage
        timing_train_opt_time = 0

        print("evaluating naive two stage.")
        tick = time.time()
        utilities_twostage, violating_usage_twostage, viorate_twostage, ratio, run_batch_total_duration = eval_perf_batch(
            y_test, preds_test, req_params_test, T, args.opt, time_lim=args.opt_time, display=True, nn_device=nn_device, parallel=args.parallel, parallel_proc=args.proc)
        tock = time.time()
        timing_test_time = tock - tick
        timing_test_cpu_time = run_batch_total_duration.total_seconds()
        
        all_text += "run time for evaluating naive two stage on test set: "
        all_text += str(run_batch_total_duration)
        all_text += "\n"

        util_ratios_twostage = np.array(utilities_twostage)/utilities_True
        print(ratio)
        print("aver uti ratio:", np.mean(util_ratios_twostage))
        print("std uti ratio:", np.std(util_ratios_twostage))
        print("aver viorate:", np.mean(viorate_twostage))
        print("std viorate:", np.std(viorate_twostage))
        all_lists.append(['TwoStage_Utilities'] + utilities_twostage)
        all_lists.append(['TwoStage_UtiRatios'] + list(util_ratios_twostage))
        all_lists.append(['TwoStage_VioRates'] + viorate_twostage)
        
        test_result_utilities = utilities_twostage
        test_result_utilities_ratio = util_ratios_twostage
        test_result_viorate = viorate_twostage

    # Skipping the RobustOpt part
    '''
    """
    RobustOpt: estimate model uncertainty and calculate capacity cutdown
    """
    # estimate the model's uncertainty over the whole training set.
    E = y_train - preds_train
    # calculate the fitted um attributes for each timestamp
    um_attribs = []
    for t in range(E.shape[1]):
        errors_t = E[:, t]
        um = GmmModel(tol=1e-10, max_components=100)
        um.fit(errors_t)
        epsilon = um.get_distribution()
        mu = epsilon.exp  # error expectation
        stdev = epsilon.std  # error standard deviation
        um_attribs.append((mu, stdev))
    # calculate the capacity cutdowns for each p
    p_s = np.arange(0.05, 0.55, 0.05)
    delta_s = []
    for p in p_s:
        delta = []    # capacity cutdowns for this p; array of length T.
        for t in range(E.shape[1]):
            mu, stdev = um_attribs[t]
            delta.append(-norm.ppf(p)*stdev-mu)
        delta_s.append(np.array(delta))

    for j in range(len(p_s)):
        p = p_s[j]
        print("evaluating RobustOpt with p =", p, "- progress:")
        preds_Ro = preds_test-delta_s[j]
        if not (preds_Ro > 0).all():
            print("[WARNING]: RO predicts <0 capacities on test set. Setting negative predictions to 0.01.")
            preds_Ro[preds_Ro < 0] = 0.01
        utilities_Ro, violating_usage_Ro, viorate_Ro, _, run_batch_total_duration = eval_perf_batch(
            y_test, preds_Ro, req_params_test, T, args.opt, time_lim=30, display=True, nn_device=nn_device, parallel=args.parallel, parallel_proc=args.proc)
        all_text += "run time for evaluating RobustOpt with p=" + str(p)+ " on test set: "
        all_text += str(run_batch_total_duration)
        all_text += "\n"
        util_ratios_Ro = np.array(utilities_Ro) / utilities_True
        print("aver uti ratio:", np.mean(util_ratios_Ro))
        print("std uti ratio:", np.std(util_ratios_Ro))
        print("aver viorate:", np.mean(viorate_Ro))
        print("std viorate:", np.std(viorate_Ro))
        all_lists.append(["RobustOpt-p="+str(p)+"_Utilities"] + utilities_Ro)
        all_lists.append(["RobustOpt-p="+str(p)+"_UtiRatios"] + list(util_ratios_Ro))
        all_lists.append(["RobustOpt-p="+str(p)+"_VioRates"] + viorate_Ro)
    '''

    if args.if_optnet == 1:
        tick = time.time()
        net.fit_optnet(X1_train/data_scaler, req_params_train, y_train/data_scaler, batch_size=10,
            total_iteration=optnet_iterations, vio_regularity=optnet_vio_regularity, req_core_scaledown=data_scaler)
        tock = time.time()
        timing_train_opt_time = tock - tick
        all_text += "run time for optnet fitting on training set: "
        all_text += str(timing_train_opt_time)
        all_text += "\n"

        # evaluation
        preds_test = net.predict(X1_test/data_scaler) * data_scaler
        if not (preds_test > 0).all():
            print("[WARNING]: optnet predicts <0 capacities on test set. Setting negative predictions to 0.01.")
            preds_test[preds_test < 0] = 0.01

        tick = time.time()
        utilities_optnet, violating_usage_optnet, viorate_optnet, _, run_batch_total_duration = eval_perf_batch(
            y_test, preds_test, req_params_test, T, args.opt, time_lim=args.opt_time, display=True, nn_device=nn_device, parallel=args.parallel, parallel_proc=args.proc)
        tock = time.time()
        timing_test_time = tock - tick
        timing_test_cpu_time = run_batch_total_duration.total_seconds()

        all_text += "run time for optnet evaluation on test set: "
        all_text += str(run_batch_total_duration)
        all_text += "\n"
        util_ratios_optnet = np.array(utilities_optnet) / utilities_True
        print("aver uti ratio:", np.mean(util_ratios_optnet))
        print("std uti ratio:", np.std(util_ratios_optnet))
        print("aver viorate:", np.mean(viorate_optnet))
        print("std viorate:", np.std(viorate_optnet))
        all_lists.append(["OptNet_Utilities"] + utilities_optnet)
        all_lists.append(["OptNet_UtiRatios"] + list(util_ratios_optnet))
        all_lists.append(["OptNet_VioRates"] + viorate_optnet)

        test_result_utilities = utilities_optnet
        test_result_utilities_ratio = util_ratios_optnet
        test_result_viorate = viorate_optnet

    print("writing results to file --")
    import csv
    with open('exp_results/'+file_name+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in zip(*all_lists):
            writer.writerow(row)
    #print(all_text)
    with open('exp_results/'+file_name+'.txt', 'w', encoding='utf-8') as f:
        f.write(all_text)

    with open('exp_results/'+file_name + '.aggregate.csv', 'w', encoding='utf-8') as f:
        f.write(",".join([
            'branch',
            'data_name_str',
            'd_factor',
            'model_name',
            'if_optnet',
            'opt_time',

            'train_time',
            'pred_time',
            'train_opt_time',
            'test_time',
            'test_cpu_time',
            
            'aver_utilities',
            'aver_util_ratio',
            'aver_viorate',
            'std_utilities',
            'std_util_ratio',
            'std_viorate',
        ]) + "\n")
        f.write(",".join([
            branch,
            data_name_str,
            str(d_factor),
            model_name_str,
            str(args.if_optnet),
            str(args.opt_time), # time limit for optimization

            str(timing_train_time), # train time
            str(timing_pred_time), # pred time
            str(timing_train_opt_time), # optimization time 
            str(timing_test_time), # optimization time
            str(timing_test_cpu_time), # optimization time

            str(np.mean(test_result_utilities)), # aver_utilities
            str(np.mean(test_result_utilities_ratio)), # aver_util_ratio
            str(np.mean(test_result_viorate)), # aver_viorate

            str(np.std(test_result_utilities)), # std_utilities
            str(np.std(test_result_utilities_ratio)), # std_util_ratio
            str(np.std(test_result_viorate)), # std_viorate
        ]) + "\n")
    
    if args.parallel:
        executor = get_parallel_executor(args.proc)
        executor.shutdown()
