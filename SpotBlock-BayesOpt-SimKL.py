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
from utils import get_sim_data_downscale_ratio, get_capa_scaledown, load_sim_capa_stat
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
    typeid = args.typeid
    d_factor = args.d_factor
    model_name = args.model_name
    if_optnet = args.if_optnet
    optnet_vio_regularity = args.optnet_vio_regularity
    optnet_iterations = args.optnet_iterations
    p = args.p          # violation threshold

    on_gpu = torch.cuda.is_available() if args.use_gpu is None else args.use_gpu 
    if args.parallel and on_gpu:
        print("[Warnning] Cannnot use GPU with parallel")
        on_gpu=False
    nn_device = torch.device('cuda') if on_gpu else torch.device('cpu')

    file_name = 'BayesOpt_'+branch+"_"+typeid+"_"+str(d_factor)+"_"+model_name+"_"+str(if_optnet)+"_"+str(args.p)
    # generate filename to write results to
    if if_optnet == 1:
        file_name = file_name+"_"+str(optnet_vio_regularity)+"_"+str(optnet_iterations)
    if args.use_sim:
        file_name = file_name+"_sim_" + str(args.use_sim_size) + '_' + str(args.use_sim_index)

    # load capacity and requests data
    print("loading data --")
    capacities, requests = load_data('data/', branch, typeid, use_sim_capacity=args.use_sim,
                                     sim_size=args.use_sim_size, sim_index=args.use_sim_index)

    capacities, true_means, true_stds = load_sim_capa_stat('data/', args.use_sim_size, args.use_sim_index, len(capacities))
    capacities, true_means, true_stds = np.array(capacities), np.array(true_means), np.array(true_stds)

    print("loading complete.")
    print("totally", len(capacities), "hours of capacity data.")
    print("total number of requests:", len(requests))

    # set pre-tuned default factors
    request_thres = 1e4 * d_factor
    if branch == 'Azure2019':
        request_thres = 0
    if branch == 'Azure2017':
        request_thres = 0

    default_capacity_scaledown = get_capa_scaledown(branch, typeid, args.use_sim, args.use_sim_size)

    # scale and downsample
    request_downsampling_factor = d_factor  # ratio of requests down-sampling
    capacity_scale_factor = default_capacity_scaledown*d_factor     # factor for scaling down the capacities
    """
    if args.use_sim:
        # for simulated data, need to modify the d-factor so as to fit the capacity
        request_downsampling_factor = get_sim_data_downscale_ratio(
            args.branch, args.typeid, args.use_sim_size) * d_factor
        capacity_scale_factor = 1
    """

    print("scaling data ---")
    print("capacity_scale_factor:", capacity_scale_factor)
    print("request_downsampling_factor:", request_downsampling_factor)
    scaled_capacities, DownSampledRequests = scale_and_downsample(
        capacities, requests, capacity_scale_factor, request_downsampling_factor)

    # scale also the mean/std
    true_means /= capacity_scale_factor
    true_stds /= capacity_scale_factor

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
    data_scaler = max_X1 / 10

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
    # ----------------------------- begin model training & evaluation ----------------------------

    def KL_div(normal1, normal2):
        # KL-dev of two normal distibutions
        mu1, sigma1 = normal1
        mu2, sigma2 = normal2
        return -1/2 + np.log(sigma2/sigma1) + (sigma1**2+(mu1-mu2)**2)/(2*sigma2**2)

    def KL_div_batch(mus_1, stds_1, mus_2, stds_2):
        assert len(mus_1)==len(stds_1)==len(mus_2)==len(stds_2)
        total = 0
        for i in range(len(mus_1)):
            mu1,sigma_1,mu2,sigma_2 =  mus_1[i],stds_1[i],mus_2[i],stds_2[i]
            total += KL_div((mu1,sigma_1),(mu2,sigma_2))
        return total

    def bayesOpt_objFunc(prediction_model, uncertainty_model, lambda_p=1e4):
        # calculate the models' performance on the **training set** and obtain a final score
        prediction_model.fit(X1_train/data_scaler, y_train/data_scaler)
        preds_train = prediction_model.predict(X1_train/data_scaler)*data_scaler

        E = y_train - preds_train
        # calculate the fitted um attributes for each timestamp
        delta = []  # capacity cutdowns; array of length T.
        for t in range(E.shape[1]):
            errors_t = E[:, t]
            uncertainty_model.fit(errors_t)
            epsilon = uncertainty_model.get_distribution()
            mu = epsilon.exp  # error expectation
            stdev = epsilon.std  # error standard deviation
            delta.append(-norm.ppf(p) * stdev - mu)
        delta = np.array(delta)

        # evaluate the RobustOpt solution's performance on the training set
        utilities, violating_usage, viorate, _, run_batch_total_duration = eval_perf_batch(
            y_train, preds_train-delta, req_params_train, T, args.opt, time_lim=30, display=False, parallel=args.parallel, parallel_proc=args.proc)
        aver_utilities = np.mean(utilities)
        aver_viorate = np.mean(viorate)

        # note: the goal is to maximize this output
        return aver_utilities - lambda_p*max(aver_viorate-p, 0)

    def evaluate_BO_testset(net_final, um_final):
        # evaluate the model on the test set
        net_final.fit(X1_train/data_scaler, y_train/data_scaler)
        preds_train = net_final.predict(X1_train/data_scaler)*data_scaler
        preds_test = net_final.predict(X1_test/data_scaler)*data_scaler
        E = y_train - preds_train       # the errors on the training set
        um_attribs = []
        delta = []  # capacity cutdowns; array of length T.
        for t in range(E.shape[1]):
            errors_t = E[:, t]
            um_final.fit(errors_t)
            epsilon = um_final.get_distribution()
            mu = epsilon.exp  # error expectation
            stdev = epsilon.std  # error standard deviation
            um_attribs.append([mu, stdev])
            delta.append(-norm.ppf(p) * stdev - mu)
        delta = np.array(delta)
        um_attribs = np.array(um_attribs)
        # evaluate the performance on the **TEST** set
        utilities, violating_usage, viorate, _, run_batch_total_duration = eval_perf_batch(
            y_test, preds_test-delta, req_params_test, T, args.opt, time_lim=30, display=False, parallel=args.parallel, parallel_proc=args.proc)
        aver_utilities = np.mean(utilities)
        aver_viorate = np.mean(viorate)

        # KL-div from true distributions
        KL_total = 0
        for i in range(len(preds_test)):
            # the timestamp window corresponding to original capacity series
            T1 = i+num_train_samples+history_lookback
            T2 = T1+T
            # find the ground truth distributions
            temp_truth_mu_s = true_means[T1:T2]
            temp_truth_std_s = true_stds[T1:T2]

            # find the predicted distributions
            temp_pred_mu_s = preds_test[i] + um_attribs[:,0] # predicted+error's mean
            temp_pred_std_s = um_attribs[:,1] # error's std
            KL_total += KL_div_batch(temp_pred_mu_s,temp_pred_std_s,temp_truth_mu_s,temp_truth_std_s)

        return aver_utilities, aver_viorate, KL_total

    # some verifications: choose a prediction model and a uncertainty model
    print("using model", model_name)
    if model_name == "FCNet":
        from grad_models import ReluFCNet
        net = ReluFCNet(input_size=X1_train.shape[1], hid1=100, hid2=50, pred_size=T, T=T,
                        SGD_params=[1e-4, 0.9], nn_device=nn_device)
        BayesOpt_selector = BO_main(FCNet_bo_config, T, object_function=bayesOpt_objFunc, input_size=X1_train.shape[1], max_iter=args.bo_iter)
    elif model_name == "LstmNet":
        from grad_models import LstmNet
        net = LstmNet(input_feature_dim=1, output_size=T, T=T, hid_size=100, step_size=1e-3,
                      nn_device=nn_device)
        BayesOpt_selector = BO_main(TSDec_bo_config, T, bayesOpt_objFunc, max_iter=args.bo_iter)
    elif model_name == "LinearFit":
        from nongrad_models.LinearFitModel import LinearFitModel
        net = LinearFitModel(T=T, latest_n=60)
        BayesOpt_selector = BO_main(LinearFit_bo_config, T, bayesOpt_objFunc, max_iter=args.bo_iter)
    elif model_name == "TSDec":
        from nongrad_models.TSDecModel import TSDecModel
        net = TSDecModel(T=T, decompose_length=360, trend_length=12, residual_length=360,
                         time_series_frequency=12)
        BayesOpt_selector = BO_main(TSDec_bo_config, T, bayesOpt_objFunc, max_iter=args.bo_iter)
    else:
        assert False
    um = GmmModel(tol=1e-10, max_components=100)
    train_eval_score_before = bayesOpt_objFunc(net, um)
    aver_utilities_before, aver_viorate_before, KL_before = evaluate_BO_testset(net, um)

    print("BO starts")
    # find BO-optimized models
    tick = time.time()
    net_final, um_final = BayesOpt_selector.run()
    tock = time.time()

    train_eval_score_after = bayesOpt_objFunc(net_final, um_final)
    print("[TRAIN SET] evaluation score before BO: ", train_eval_score_before,
          " and after BO: ", train_eval_score_after)

    aver_utilities_after, aver_viorate_after, KL_after = evaluate_BO_testset(net_final, um_final)
    print("[TEST SET] aver utilities and aver viorate before BO:")
    print(aver_utilities_before, aver_viorate_before)
    print("[TEST SET] aver utilities and aver viorate after BO:")
    print(aver_utilities_after, aver_viorate_after)
    print("[TEST SET] KL-div before/after BO:")
    print(KL_before, KL_after)

    with open('exp_results/'+file_name+'.txt', 'w', encoding='utf-8') as f:
        f.write(f"{file_name}\n")
        f.write(f"[Train set]:\n")
        f.write(f" evaluation score before BO: {train_eval_score_before}\n")
        f.write(f" evaluation score after BO: {train_eval_score_after}\n")
        f.write(f"[Test set]:\n")
        f.write(f" aver utilities and aver viorate before BO: {aver_utilities_before}, {aver_viorate_before}\n")
        f.write(f" aver utilities and aver viorate after BO: {aver_utilities_after}, {aver_viorate_after}\n")
        f.write(f" KL-div before/after BO: {KL_before}, {KL_after}\n")

    if args.parallel:
        executor = get_parallel_executor(args.proc)
        executor.shutdown()
