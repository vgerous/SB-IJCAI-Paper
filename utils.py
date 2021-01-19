import numpy as np
import pandas as pd

def load_simulated_capacity(data_path, size, index, timestamp_length):
    data_size_folder = {
        'large': 'Large_size_simulate_data',
        'middle': 'Middle_size_simulate_data',
        'small': 'Small_size_simulate_data',
    }
    data_file_name = {
        'large': str(index) + '_large_simulated_capacity_with_truth.csv',
        'middle': str(index) + '_simulated_capacity_with_truth.csv',
        'small': str(index) + '_small_simulated_capacity_with_truth.csv',
    }
    path = data_path + data_size_folder[size.lower()] + '/' + data_file_name[size.lower()]
    data = pd.read_csv(path, header=0)
    return data.y[0:timestamp_length].to_list()

def load_sim_capa_stat(data_path, size, index, timestamp_length):
    data_size_folder = {
        'large': 'Large_size_simulate_data',
        'middle': 'Middle_size_simulate_data',
        'small': 'Small_size_simulate_data',
    }
    data_file_name = {
        'large': str(index) + '_large_simulated_capacity_with_truth.csv',
        'middle': str(index) + '_simulated_capacity_with_truth.csv',
        'small': str(index) + '_small_simulated_capacity_with_truth.csv',
    }
    path = data_path + data_size_folder[size.lower()] + '/' + data_file_name[size.lower()]
    data = pd.read_csv(path, header=0)
    return data.y[0:timestamp_length].to_list(), data.Mean[0:timestamp_length].to_list(), data.Sigma[0:timestamp_length].to_list()


def load_data(data_path, branch, typeid, use_sim_capacity=False, sim_size='Large', sim_index=1):
    assert branch in ['Default', 'Azure2019', 'Azure2017']
    if branch == 'Default':
        assert typeid in ['65','64','41','42','m1','m2','m3','m4']

        data_path_real = data_path+'Mixed/' if typeid.startswith('m') else data_path

        capacity_data = np.array(pd.read_csv(data_path_real+"capacity_data_Default.csv", header=0))
        capacities = []
        for entry in capacity_data:
            timestamp, typestr, numcores = entry
            if typestr == typeid+'str':
                capacities.append(numcores)
        if use_sim_capacity:
            capacities = load_simulated_capacity(data_path, sim_size, sim_index, len(capacities))

        jobrequests_data = np.array(pd.read_csv(data_path_real + "JobRequests_Default.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == typeid:
                demand_core_scale_factor = 40 if use_sim_capacity else 1
                requests.append([demand_core * demand_core_scale_factor, duration, earliest_time, latest_time])
        return np.array(capacities), np.array(requests)

    if branch == 'Azure2019':
        assert typeid in ['t1','t2','t3','t5','t6','t7','m1','m2','m3','m4']

        data_path_real = data_path+'Mixed/' if typeid.startswith('m') else data_path

        capacity_data = np.array(pd.read_csv(data_path_real + "capacity_data_Azure2019.csv", header=0))
        capacities = []
        for entry in capacity_data:
            timestamp, typestr, numcores = entry
            if typestr == typeid + 'str':
                capacities.append(numcores)

        if use_sim_capacity:
            capacities = load_simulated_capacity(data_path, sim_size, sim_index, len(capacities))

        jobrequests_data = np.array(pd.read_csv(data_path_real + "JobRequests_Azure2019.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == typeid:
                requests.append([demand_core, duration, earliest_time, latest_time])
        return np.array(capacities), np.array(requests)

    if branch == 'Azure2017':
        assert typeid in ['t1','t2','t3','t5','t6','m1','m2','m3','m4']  # 't7' is banned here
        
        data_path_real = data_path+'Mixed/' if typeid.startswith('m') else data_path

        capacity_data = np.array(pd.read_csv(data_path_real + "capacity_data_Azure2017.csv", header=0))
        capacities = []
        for entry in capacity_data:
            timestamp, typestr, numcores = entry
            if typestr == typeid + 'str':
                capacities.append(numcores)

        if use_sim_capacity:
            capacities = load_simulated_capacity(data_path, sim_size, sim_index, len(capacities))

        jobrequests_data = np.array(pd.read_csv(data_path_real + "JobRequests_Azure2017.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == typeid:
                requests.append([demand_core, duration, earliest_time, latest_time])
        return np.array(capacities), np.array(requests)


def load_GroundTruth_utilities(GroundTruth_path, branch, typeid, d_factor, if_sim, sim_size, sim_index):
    file_name = branch+"_"+typeid+"_"+str(d_factor)
    if if_sim:
        file_name = file_name + "_sim_" + str(sim_size) + '_' + str(sim_index)
    return np.array(pd.read_csv(GroundTruth_path + file_name + '_GroundTruth.csv', header=0)).reshape(-1)

def scale_and_downsample(capacities, requests, capacity_scale_factor, request_downsampling_factor, seed=0):
    """
    params: as the name suggests
    returns: scaled capacites, downsampled requests
    """
    num_requests = round(request_downsampling_factor*requests.shape[0])
    np.random.seed(seed)
    rand_ind = np.random.choice(requests.shape[0], num_requests, replace=False)
    return capacity_scale_factor*capacities, requests[rand_ind]


def filter_requests(T1, T2, requests):
    """
    filter the requests in window T1-T2 from requests. T1 is subtracted from earliest/latest start time to pretend T1=0.
    """
    filtered_requests = []
    for request_entry in requests:
        cores, duration, earliest_time, latest_time = request_entry
        if (duration+earliest_time > T2) or (latest_time < T1):
            # this request is out of the window
            continue
        # what's left are potentially-deployable VM requests within considered time range. Align to T1=0.
        filtered_requests.append([cores, duration, max(earliest_time, T1)-T1, min(latest_time, T2-duration)-T1])
    return np.array(filtered_requests)


def requests2params(requests):
    """
    Extract parameters for the Spot Block problem from requests data.
    returns:
        N: number of requests (VMs)
        c: requested cores for each VM
        d: requested time duration for each VM
        e: requested earliest start time for each VM
        l: requested latest start time for each VM
    """
    N = requests.shape[0]                # number of requests (or number of VMs)
    if N == 0:
        return 0, np.array([]), np.array([]), np.array([]), np.array([])
    c = requests[:, 0]                    # requested cores for each VM
    d = requests[:, 1].astype(int)        # requested time duration for each VM
    e = requests[:, 2].astype(int)        # requested earliest start time for each VM
    l = requests[:, 3].astype(int)        # requested latest start time for each VM
    return N, c, d, e, l


def cal_demand_and_profit(decisions,N,T,c,d):
    """
    calculate the total demanded cores (array of length T) and profit, given the decisions.
    params as in the problem formulation.
    """
    demand_cores_matrix = np.zeros([N, T])
    for i in range(N):
        if decisions[i] == None:
            continue
        temp_cores = np.zeros(T)
        temp_cores[decisions[i]:(decisions[i]+d[i])] = c[i]
        demand_cores_matrix[i]=temp_cores
    total_demand_cores = np.sum(demand_cores_matrix, axis=0)
    # calculate the profit for the strategy.
    profit = 0
    for i in range(len(decisions)):
        if decisions[i] != None:
            profit += (c[i]*d[i])
    return total_demand_cores,profit


def greedy_improvement(decisions,N,T,a,c,d):
    """
    greedily improve the decicions.
    params:
        N,T,a,c,d: same as in the problem formulation.
        decisions: a list of length N, the entries are the decided start times/None for the VMs (None indicates no deploy).
    returns:
        decisions: the improved decisions which induces a strategy that satisfies the capacity constraints.
        total_demand_cores: a 1d array with length T, each entry is the occupied capacity by the IMPROVED decision.
        profit: the profit (objective value) by the IMPROVED decision.
    """
    while True:
        # calculate the number of cores occupied for each time and each VM
        demand_cores_matrix = np.zeros([N, T])   # (i,t) th entry is VM i's core demand at time t.
        for i in range(N):
            if decisions[i] == None:
                continue
            temp_cores = np.zeros(T)
            temp_cores[decisions[i]:(decisions[i]+d[i])] = c[i]
            demand_cores_matrix[i]=temp_cores
        total_demand_cores = np.sum(demand_cores_matrix, axis=0)
        remaining_capacities = a - total_demand_cores
        violating_timesteps = np.argwhere(remaining_capacities < 0).reshape(-1)
        if len(violating_timesteps) == 0:
            # no violating timestep - break the loop
            break
        # pick the one timestep with most serious violation
        MostSeriousViolTime = np.argmin(remaining_capacities)
        # calculate how many cores for this timestep need to be removed
        thres = -remaining_capacities[MostSeriousViolTime] 
        # now remove the employments from biggest occupation to smallest occupation at this timestep
        column_vec = demand_cores_matrix[:, MostSeriousViolTime]   # vector of occupied cores for each VM at this timestep.
        # sort the VM's according to their occupied cores at this timestep, from big->small
        l_temp = [(i, column_vec[i]) for i in range(len(column_vec))]
        l_temp.sort(key=lambda var: var[1], reverse=True)
        # now undeploy till thres is reached.
        accumulated_sum = 0
        remove_index = []
        for (i, num_cores_occupied) in l_temp:
            if accumulated_sum < thres:
                remove_index.append(i)
                accumulated_sum += num_cores_occupied
            else:
                break
        for i in remove_index:
            decisions[i] = None
    # assert that the constraints are satisfied.
    total_demand_cores, profit = cal_demand_and_profit(decisions,N,T,c,d)
    usage_ratio = total_demand_cores / a
    assert (usage_ratio < 1).all()   # all usage-ratio should <1.
    return decisions, total_demand_cores, profit

def parse_arguments():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('branch', type=str, help='which branch of data to use')
    parser.add_argument('typeid', type=str, help='typeid of VMs considered')
    parser.add_argument('d_factor', type=float, help='ratio of requests down-sampling')
    parser.add_argument('model_name', nargs='?', type=str, help='prediction model')
    parser.add_argument('if_optnet', nargs='?', type=int, help='1/0 indicating whether use optnet option')
    parser.add_argument('optnet_vio_regularity', nargs='?', type=float, default=50, help='')
    parser.add_argument('optnet_iterations', nargs='?', type=int, default=10, help='')
    parser.add_argument('--use-sim', type=bool, default=False, help='whether to use simulated capacity')
    parser.add_argument('--use-sim-size', type=str, default='large', help='category of simulated capacity')
    parser.add_argument('--use-sim-index', type=int, default=1, help='file index of simulated capacity')
    parser.add_argument('--use-gpu', type=bool, default=None, help='whether to use gpu')
    parser.add_argument('--p', type=float, default=0.05, help='violation threshold p (BayesOpt only)')
    parser.add_argument('--parallel', type=bool, default=False, help='Use process-poll parallelism')
    parser.add_argument('--proc', type=int, default=6, help='Number of processes')
    parser.add_argument('--override', type=bool, default=False, help='whether to override existing cache (Cache only)')
    parser.add_argument('--opt', type=str, default='MIP', help='Optimization to use: MIP, Heuristic')
    parser.add_argument('--bo-iter', type=int, default=10, help='Iterations in Bayesian Optimization')

    try:
        args = parser.parse_args()
        print('arguments: ')
        print(args)
        return args
    except:
        parser.print_help()
        exit()

def get_sim_data_downscale_ratio(branch, typeid, sim_size):
    mapping = {
        'Azure2019': {
            'small': {'t1': 0.04, 't2': 0.03, 't3': 0.1, 'm1': 0.02, 'm2': 0.02, 'm3': 0.012},
            'middle': {'t1': 0.25, 't2': 0.15, 't3': 0.6, 'm1': 0.15, 'm2': 0.15, 'm3': 0.1},
            'large': {'t1': 0.5, 't2': 0.3, 'm1': 0.5, 'm2': 0.28, 'm3': 0.18 },
        },
        'Azure2017': {
            'small': {'m1':0.04, 'm2':0.05, 'm3':0.035},
            'middle': {'m1':0.22, 'm2':0.22, 'm3':0.2},
            'large': {'m1':0.45, 'm2':0.48, 'm3':0.3},
        },
        'Default': {
            'small': {'m1':0.025, 'm2':0.06, 'm3':0.04},
            'middle': {'m1':0.12, 'm2':0.3, 'm3':0.2},
            'large': {'m1':0.23, 'm2':0.6, 'm3':0.42},
        },
    }
    dic = mapping
    if branch not in dic.keys():
        return 1.0
    dic = dic[branch]
    if sim_size not in dic.keys():
        return 1.0
    dic = dic[sim_size]
    if typeid not in dic.keys():
        print(typeid,dic.keys())
        return 1.0
    return dic[typeid]

def scale_dict_vals(dictionary, factor):
    # recursively scale all values in dictionary by factor
    for key in dictionary.keys():
        if type(dictionary[key]) == dict:
            dictionary[key] = scale_dict_vals(dictionary[key], factor)
        else:
            dictionary[key] = dictionary[key] * factor
    return dictionary


def get_capa_scaledown(branch, typeid, if_sim, sim_size):
    if not if_sim:
        # real data params
        mapping = {
            'Default': {
                '64': 0.15, '65': 0.06, '42': 0.03, '41': 0.03,
                'm1': 0.04, 'm2': 0.06, 'm3': 0.05, 'm4': 0.05
            },
            'Azure2019': {
                't1': 0.055, 't2': 0.07, 't3': 0.013, 't5': 0.05, 't6': 0.015, 't7': 0.1,
                'm1': 0.04, 'm2': 0.025, 'm3': 0.06, 'm4': 0.05
            },
            'Azure2017': {
                't1': 0.014, 't2': 0.032, 't3': 0.015, 't5': 0.05, 't6': 0.023,
                'm1': 0.03, 'm2': 0.02, 'm3': 0.03, 'm4': 0.04
            },
        }
        dic = mapping
        if branch not in dic.keys():
            assert False
        dic = dic[branch]
        if typeid not in dic.keys():
            assert False
        return dic[typeid]

    # sim data params
    sim_middle_dict = {
        'Default': {
            '64': 1.5, '65': 2.0, '42': 3.0, '41': 1.0,
            'm1': 6.0, 'm2': 3.0, 'm3': 2.5, 'm4': 6.0
        },
        'Azure2019': {
            't1': 3.5, 't2': 6.0, 't3': 1.3, 't5': 5.0, 't6': 1.5, 't7': 4.0,
            'm1': 8.0, 'm2': 2.5, 'm3': 6.0, 'm4': 7.0
        },
        'Azure2017': {
            't1': 1.0, 't2': 2.3, 't3': 1.7, 't5': 2.0, 't6': 0.2,
            'm1': 6.0, 'm2': 3.5, 'm3': 3.0, 'm4': 4.0
        },
    }

    dic = {
        'small': scale_dict_vals(sim_middle_dict, 0.2),
        'middle': sim_middle_dict,
        'large': scale_dict_vals(sim_middle_dict, 2)
    }
    if sim_size not in dic.keys():
        assert False
    dic = dic[sim_size]
    if branch not in dic.keys():
        assert False
    dic = dic[branch]
    if typeid not in dic.keys():
        assert False
    return dic[typeid]