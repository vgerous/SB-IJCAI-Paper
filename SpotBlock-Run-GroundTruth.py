#!/usr/bin/env python
# coding: utf-8

import warnings
import numpy as np
from scipy.stats import norm
import time
import torch
from os import path
# import matplotlib.pyplot as plt

# local libraries & helper functions
from utils import load_data, scale_and_downsample, get_sim_data_downscale_ratio, parse_arguments, get_capa_scaledown
from qpth_utils import collect_ML_data, eval_perf_batch, get_parallel_executor


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # retrieve params
    args = parse_arguments()
    # retrieve params
    branch = args.branch
    typeid = args.typeid
    d_factor = args.d_factor

    on_gpu = torch.cuda.is_available() if args.use_gpu is None else args.use_gpu 
    nn_device = torch.device('cuda') if on_gpu else torch.device('cpu')

    file_name = branch+"_"+typeid+"_"+str(d_factor)
    # generate filename to write results to
    if args.use_sim:
        file_name = file_name+"_sim_"+str(args.use_sim_size)+'_'+str(args.use_sim_index)

    ground_truth_file_name = 'exp_results/GroundTruth_results/' + file_name + '_GroundTruth.csv'

    if path.exists(ground_truth_file_name) and not args.override:
        print('GroundTruth is already available at ' + file_name)
        exit()

    all_lists = []  # used for writing results to a .csv file
    # load capacity and requests data
    print("loading data --")
    capacities, requests = load_data('data/', branch, typeid, use_sim_capacity=args.use_sim,
                                     sim_size=args.use_sim_size, sim_index=args.use_sim_index)
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

    #plt.plot(scaled_capacities)
    #plt.show()

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
    # calculate maximum utility & deploy ratio using ground truth capacity via MIP;
    # the ratios can reflect how challenging the instances are.

    print("evaluating ground truth train set performance. progress:")
    _, _, _, deploy_ratios_train, run_batch_total_duration = eval_perf_batch(
        y_train, y_train, req_params_train, T, 'MIP', time_lim=1e4, display=True, nn_device=nn_device, parallel=args.parallel, parallel_proc=args.proc)
    print("train aver deploy ratio:", np.mean(deploy_ratios_train))
    print("train std deploy ratio:", np.std(deploy_ratios_train))
    #print(deploy_ratios_train)


    print("evaluating ground truth test set performance. progress:")
    utilities_True, _, _, deploy_ratios, run_batch_total_duration = eval_perf_batch(
        y_test, y_test, req_params_test, T, 'MIP', time_lim=1e4, display=True, nn_device=nn_device, parallel=args.parallel, parallel_proc=args.proc)
    print("test aver deploy ratio:", np.mean(deploy_ratios))
    print("test std deploy ratio:", np.std(deploy_ratios))
    #print(deploy_ratios)
    all_lists.append(['True_Utilities'] + utilities_True)

    # see the employ ratio curve
    #plt.plot(deploy_ratios_train+deploy_ratios)
    #plt.show()
    # plt.plot(deploy_ratios_train+deploy_ratios)
    # plt.show()

    print("writing results to file --")
    import csv
    with open(ground_truth_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in zip(*all_lists):
            writer.writerow(row)
            
    if args.parallel:
        executor = get_parallel_executor(args.proc)
        executor.shutdown()
