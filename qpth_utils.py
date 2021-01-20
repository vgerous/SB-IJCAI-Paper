from os import system
import numpy as np
from scipy.sparse import lil_matrix
import torch
import gurobipy as gp
from gurobipy import GRB
import time
import os
import subprocess
import datetime
from concurrent.futures import ProcessPoolExecutor

from utils import filter_requests, requests2params, cal_demand_and_profit, greedy_improvement
from qpth_local.qp import QPFunction, QPSolvers


def transform_qpth(T,N,c,d,e,l,epsilon=2e-3,sparse=True):
    """
    transform the problem formulation into standard QP form. 
    Params:
        T,N,c,d,e,l: same as in the Spot Block problem formulation.
        epsilon: coefficient for the quadratic term (Q=epsilon*I)
        sparse: whether use sparse matrix implementation.
    Returns:
        epsilon,p: objective = 1/2 epsilon*x^Tx  +  p^T x  (using epsilon instead of Q to save memory usage.)
        G,h: constraint is Gx <= h
    VERY IMPORTANT:
        h[:T] is the slot for total capacity constraints (a in SB formulation) and is empty in the returned value, 
        need to be filled with predicted/real capacities outside this function.
    """
    # the dimension of the decision variables for each VM
    dims = l-e+1
    # total number of decision variables
    total_dim = int(np.sum(dims))

    # parameters in the objective
    p = np.zeros(total_dim)
    for i in range(N):
        index = np.sum(dims[:i])
        p[index: index+dims[i]] = -c[i]*d[i]
    
    # parameters in the constraints
    if sparse:
        # use sparse matrix implementation
        G = lil_matrix((T+N, total_dim))
    else:
        G = np.zeros((T+N, total_dim))

    h = np.zeros(T + N)
    # 1. capacity constraint. details see problem formulation
    for t in range(T):
        x = np.zeros(total_dim)
        for i in range(N):
            index = np.sum(dims[:i])
            left = max(e[i],t-d[i]+1)
            right = min(t, l[i])
            left -= e[i]
            right -= e[i]
            if right >= left:
                x[index+left:index+right+1] = c[i]
        G[t] = x
    # h[:T] should be the capacities; NEED to be filled LATER outside this function.

    # 2. append nen-negativity constraint: automatic due to the local adaptation of CVXPY solver option of qpth
    # 3. less-than-1-constraints
    for i in range(N):
        x = np.zeros(total_dim)
        index = np.sum(dims[:i])
        G[T+i, index: index+dims[i]] = 1
    h[T:T+N] = 1
    # number of inequality constraints (without non-negative constraints): G.shape[0]
    return epsilon,p,G,h


def GetDecisionFromSol(sol, N, T, e, l):
    """
    induce from the MIP/QP solution the decisions.
    For each VM the start time would be set at the time with largest value (if the largest > 1e-2).
    params:
        sol: the MIP/QP solution.
        N,T,e,l: same as in problem formulation
    returns:
        decisions: array of length N, each the scheduled start time or None for no deploy.
        deploy ratio: number of deployed VM / total # of VM
    """
    # sol: the .
    # the decision variables; either the starting time or None (no-deploy). Currently all VMs are deployed, could be modified.
    dims = l-e+1
    total_dim = int(np.sum(dims))

    decisions = []
    # count how many employed VMs
    counter = 0
    for i in range(N):
        index = np.sum(dims[:i])
        decision_variables = sol[index:index+dims[i]]
        if np.sum(decision_variables) < 1e-2:
            decisions.append(None)
            continue
        decisions.append(e[i]+np.argmax(decision_variables))
        counter += 1

    return decisions, counter/N

def collect_ML_data(history_lookback, T, request_thres, capacities, requests):
    """
    params: as names suggest
    returns:
        data_X1: history capacities
        req_params: current requests data
        data_y: ground truth current capacities
    """
    data_X1 = []     # history capacities
    req_params = []  # current requests data
    data_y = []      # ground truth current capacities
    for t in range(len(capacities)):
        # collect history capacities and current capacities in a sliding window style
        T1, T2 = t+history_lookback, t+history_lookback+T
        if T2 >= len(capacities):
            # out of range
            break
        filtered_requests = filter_requests(T1, T2, requests)
        if len(filtered_requests) < request_thres:
            # too few requests
            break
        data_X1.append(capacities[t:T1])
        data_y.append(capacities[T1:T2])
        req_params.append(filtered_requests)

    return np.array(data_X1), req_params, np.array(data_y)

def eval_performance(a, a_hat, req_param, T, solve_method, time_limit, nn_device=torch.device('cpu')):
    """
    Evaluate the estimated capacity's performance for a time window instance.
    params:
        a: ground truth capacities
        a_hat: estimated capacities
        req_param: request param
        T: length of time window
        solve_method: 'QP+greedy'/'MIP' (feel free to add others such as Heurustic Search)
        display: print some info
    returns:
        total profit (obj value)
        number of times of capacity violation
        total violating capacities
        deploy ratio of VMs
    """
    if solve_method == 'QP+Greedy':
        """
        # retrieve QP params
        N,c,d,e,l = requests2params(req_param)
        epsilon,p,G,h = transform_qpth(T,N,c,d,e,l,sparse=False)   # qpth doesn't support sparse matrix input yet
        Q = torch.Tensor(epsilon*np.eye(G.shape[1])).to(nn_device)
        p = torch.Tensor(p).to(nn_device)
        G = torch.Tensor(G).to(nn_device)
        h = torch.Tensor(h).to(nn_device)
        null_var = torch.Tensor().to(nn_device)
        h[:T] = torch.Tensor(a_hat).to(nn_device)           # fill in the predicted capacities
        sol = QPFunction(verbose=False, solver=QPSolvers.CVXPY, nn_device=nn_device)(Q, p, G, h, null_var, null_var)[0] # solve 
        # make sure consraints are satisfied
        # assert (sol>-1e-4).all()
        # assert (torch.matmul(G, sol)-h < 1e-4).all()
        decisions, _ = GetDecisionFromSol(sol.detach().numpy(), N, T, e, l)
        #decisions, deploy_ratio = HeuristicSearch(T,N,c,d,e,l,a_hat)
        #print("decisions: ", decisions)
        decisions,total_demand_cores,profit = greedy_improvement(decisions,N,T,a_hat,c,d)
        ind = np.argwhere(total_demand_cores > a).reshape(-1) 
        return profit,len(ind),np.sum(total_demand_cores[ind]-a[ind]),None
        """
        # currently no use of this method.
        assert False

    if solve_method == 'MIP':
        tick = time.time()
        N,c,d,e,l = requests2params(req_param)
        _,p,G,h = transform_qpth(T,N,c,d,e,l,sparse=True)   # no use of quadratic term here
        h[:T] = a_hat
        tock = time.time()
        # print("transform params using", tock-tick, "seconds")

        # Create a new gurobi model
        tick = time.time()
        m = gp.Model("mip1")
        m.Params.LogToConsole = 0
        m.setParam(GRB.Param.TimeLimit, time_limit)
        # Create variables
        x = m.addMVar(G.shape[1], vtype=GRB.BINARY, name="x")
        m.setObjective(p @ x, GRB.MINIMIZE)
        m.addConstr(G @ x <= h, "c")
        m.optimize()
        sol = np.array(m.getAttr("x"))
        tock = time.time()
        # make sure that all constraints are satisfied
        # assert (sol>-1e-4).all()
        # assert (np.dot(G, sol)-h < 1e-4).all()
        # print("solve MIP using", tock-tick, "seconds")
        decisions, deploy_ratio = GetDecisionFromSol(sol, N, T, e, l)
        #decisions, deploy_ratio = HeuristicSearch(T,N,c,d,e,l,a_hat)
        #print("decisions: ", decisions)
        #print("deploy_ratio: ", deploy_ratio)
        total_demand_cores, profit = cal_demand_and_profit(decisions,N,T,c,d)
        ind = np.argwhere(total_demand_cores > a).reshape(-1) 
        return profit, len(ind), np.sum(total_demand_cores[ind]-a[ind]), deploy_ratio

    if solve_method == 'Heuristic':
        tick = time.time()
        N, c, d, e, l = requests2params(req_param)
        tock = time.time()
        # print("transform params using", tock-tick, "seconds")

        # Create a new model
        tick = time.time()
        decisions, deploy_ratio = HeuristicSearch(T,N,c,d,e,l,a_hat)
        tock = time.time()
        # make sure that all constraints are satisfied
        # assert (sol>-1e-4).all()
        # assert (np.dot(G, sol)-h < 1e-4).all()
        # print("solve Heuristic using", tock-tick, "seconds")

        # print("decisions: ", decisions)
        total_demand_cores,profit = cal_demand_and_profit(decisions,N,T,c,d)  
        ind = np.argwhere(total_demand_cores > a).reshape(-1) 
        return profit,len(ind),np.sum(total_demand_cores[ind]-a[ind]),deploy_ratio

    assert False


def eval_perf_batch_step(task):
    index, total_task, display, a_s, a_hat_s, req_params, T, solve_method, time_limt = task
    start_time = datetime.datetime.utcnow()
    if display:
        print(f"    [{index} / {total_task}] Starting task {start_time}")

    profit, num_violations, vio_capacity, deploy_ratio = eval_performance(
        a_s, a_hat_s, req_params, T, solve_method, time_limit=time_limt)
    end_time = datetime.datetime.utcnow()
    if display:
        print(f"    [{index} / {total_task}] Finish {end_time}, Duration {end_time - start_time}")

    return profit, num_violations/T, vio_capacity, deploy_ratio, end_time - start_time

shared_parallel_executor = None
def get_parallel_executor(max_workers=6):
    global shared_parallel_executor
    if shared_parallel_executor == None:
        print("Start thread pool for evaluation")
        shared_parallel_executor = ProcessPoolExecutor(max_workers=max_workers)
    return shared_parallel_executor

def eval_perf_batch(a_s, a_hat_s, req_params, T, solve_method, time_lim, display=False, nn_device=torch.device('cpu'), parallel=False, parallel_proc=6):
    """
    evaluate the performance of a batch of predicted capacities.
    params:
        a_s: real capacities (stacked in rows)
        a_hat_s: predicted capacities (stacked in rows)
        req_params: requests data (stacked in rows)
        T: length of time window
        solve_method: 'MIP'(recommended)/'QP+greedy'
        display: whether display some progress info
    returns:
        utilities: list of utilities of the obtained strategies
        vio_usage: list of total violating capacity uses
        vio_rate: list of frequencies of capacity violation
        deploy_ratios: list of deploy ratios for each instance
    """

    if parallel:
        total_task = len(a_s)
        tasks = [(
            i,
            total_task,
            display,
            a_s[i],
            a_hat_s[i],
            req_params[i],
            T,
            solve_method,
            time_lim
        ) for i in range(total_task)]

        executor = get_parallel_executor(max_workers=parallel_proc)
        results = list(executor.map(eval_perf_batch_step, tasks))

        utilities = list([r[0] for r in results])
        vio_rate = list([r[1] for r in results])
        vio_usage = list([r[2] for r in results])
        deploy_ratios = list([r[3] for r in results])
        total_duration = sum([r[4] for r in results], datetime.timedelta())
        return utilities, vio_usage, vio_rate, deploy_ratios, total_duration

    utilities = []
    vio_usage = []
    vio_rate = []
    deploy_ratios = []
    durations = []
    for i in range(len(a_s)):
        start_time = datetime.datetime.utcnow()
        if display:
            print(i, end=" ", flush=True)      # just for monitor progress
        profit, num_violations, vio_capacity, deploy_ratio = eval_performance(
            a_s[i], a_hat_s[i], req_params[i], T, solve_method, time_limit=time_lim, nn_device=nn_device)
        utilities.append(profit)
        vio_rate.append(num_violations/T)
        vio_usage.append(vio_capacity)
        deploy_ratios.append(deploy_ratio)
        end_time = datetime.datetime.utcnow()
        if display and end_time - start_time > datetime.timedelta(seconds=45):
            print(f"    [{i} / {len(a_s)}] Start {start_time}, Duration {end_time - start_time}")
        durations.append(end_time - start_time)
    if display:
        print()
    
    total_duration = sum([d for d in durations], datetime.timedelta())
    return utilities, vio_usage, vio_rate, deploy_ratios, total_duration


def HeuristicSearch(T,N,c,d,e,l,a_hat):
    from heuristic import search
    X = search(T, N, c, d, e, l, a_hat)

    decisions, counter = get_decisions_HS(np.array(X))
    #print("deploy_ratio: ", counter/N)
    return decisions, counter/N

def HeuristicSearch_cpp_exe(T,N,c,d,e,l,a_hat):
    X = np.zeros((N, T))

    nowTime = f"ex-{str(os.getpid())}-{str(datetime.datetime.utcnow().timestamp())}"
    filename = "exchange_files/" + nowTime + "_in.txt"
    f = open(filename, 'w+')
    print(T, end=' ', file=f)
    print(N, end=' ', file=f)
    for i in c:
        print(i, end=' ',file=f)
    for i in d:
        print(i, end=' ',file=f)
    for i in e:
        print(i, end=' ',file=f)
    for i in l:
        print(i, end=' ',file=f)
    for i in a_hat:
        print(i, end=' ',file=f)
    f.close()

    subprocess.run([
        './HS_cpp',
        "exchange_files/" + nowTime
    ])

    output_name = "exchange_files/" + nowTime + "_out.txt"
    with open(output_name, 'r+') as f:
        lines= f.readlines()

    row=0
    for line in lines:
        line=line.strip().split('\t')
        X[row,:]=line[:][0].split()
        row+=1

    os.remove(filename)
    os.remove(output_name)

    decisions, counter = get_decisions_HS(X)
    #print("deploy_ratio: ", counter/N)
    return decisions, counter/N

def get_decisions_HS(X):
    decision = []
    counter = 0
    for cur_raw in X:
        if sum(cur_raw) == 0:
            decision.append(None)
            continue
        elif sum(cur_raw) == 1:
            decision.append(np.argmax(cur_raw))
            counter += 1
            continue
        else:
            print("deploy on more than one time!")
            assert()

    return decision, counter