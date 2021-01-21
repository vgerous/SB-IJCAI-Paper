import numpy as np
import pandas as pd


def load_data(data_path, branch):
    assert branch in ['Default', 'Azure2019', 'Azure2017']
    if branch == 'Default':
        capacities = pd.read_csv(data_path+"Capacities_Default.csv", header=None)

        jobrequests_data = np.array(pd.read_csv(data_path + "JobRequests_Default.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == 'm1':
                requests.append([demand_core, duration, earliest_time, latest_time])
        return np.array(capacities).reshape(-1), np.array(requests)

    if branch == 'Azure2019':
        capacities = pd.read_csv(data_path + "Capacities_Azure2019.csv", header=0)

        jobrequests_data = np.array(pd.read_csv(data_path + "JobRequests_Azure2019.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == 'm3':
                requests.append([demand_core, duration, earliest_time, latest_time])
        return np.array(capacities).reshape(-1), np.array(requests)

    if branch == 'Azure2017':
        capacities = pd.read_csv(data_path + "Capacities_Azure2017.csv", header=0)

        jobrequests_data = np.array(pd.read_csv(data_path + "JobRequests_Azure2017.csv", header=0))
        requests = []
        for entry in jobrequests_data:
            typeid_int, demand_core, duration_timastamp, earliest_time, latest_time = entry
            duration = int(str(duration_timastamp).split(":")[0])
            if str(typeid_int) == 'm2':
                requests.append([demand_core, duration, earliest_time, latest_time])
        return np.array(capacities).reshape(-1), np.array(requests)


def load_GroundTruth_utilities(GroundTruth_path, branch, d_factor):
    file_name = branch+"_"+str(d_factor)
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
    parser.add_argument('d_factor', type=float, help='ratio of requests down-sampling')
    parser.add_argument('model_name', nargs='?', type=str, help='prediction model')
    parser.add_argument('if_optnet', nargs='?', type=int, help='1/0 indicating whether use optnet option')
    parser.add_argument('optnet_vio_regularity', nargs='?', type=float, default=50, help='')
    parser.add_argument('optnet_iterations', nargs='?', type=int, default=10, help='')
    parser.add_argument('--use-gpu', type=bool, default=None, help='whether to use gpu')
    parser.add_argument('--p', type=float, default=0.05, help='violation threshold p (BayesOpt only)')
    parser.add_argument('--parallel', type=bool, default=False, help='Use process-poll parallelism')
    parser.add_argument('--proc', type=int, default=6, help='Number of processes')
    parser.add_argument('--override', type=bool, default=False, help='whether to override existing cache (Cache only)')
    parser.add_argument('--opt', type=str, default='MIP', help='Optimization to use: MIP, Heuristic')
    parser.add_argument('--bo-iter', type=int, default=10, help='Iterations in Bayesian Optimization')
    parser.add_argument('--opt-time', type=int, default=30, help='Time limit for optimization (applicable to MIP)')
    parser.add_argument('--skip-uncertainity', type=bool, default=False, help='Skip uncertainity modeling for BO')

    try:
        args = parser.parse_args()
        print('arguments: ')
        print(args)
        return args
    except:
        parser.print_help()
        exit()
