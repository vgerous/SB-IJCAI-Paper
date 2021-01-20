# Predictive Scheduling under Capacity Uncertainty for Spot Block Instances inCloud Computing

The proactive spot block scheduling problem is a predict + optimize problem for  the spot block products in cloud computing platforms, where the predicted capacity guides the optimization of job scheduling and job scheduling results are leveraged to improve the prediction of capacity. 

## Package Dependencies
- **python3**, **numpy**, **torch**, **cvxpy**, **gurobi**, **scikit-learn==0.22.2.post1**
- you can run the script **gcr-setup.sh** to setup the environment.
- to support the heuristic search optimization method implemented in C++, please run the code in linux environment.

## Precache Ground Truth Results
- Before running the experiments, we need to cache the ground truth results in order to evaluate our results. 
- For precaching ground truth results, run:
```
python3 SpotBlock-Run-GroundTruth.py <data_branch> <typeid> <downSampleFactor>
```
```
Example: python3 SpotBlock-Run-GroundTruth.py Azure2019 t1 0.01
```
- data_branch: 'Default', 'Azure2019', 'Azure2017'
- typeid: 65, 64, 42, 41 for 'Default' data branch; 't1','t2','t3','t5','t6','t7' for 'Azure2019' branch; 't1','t2','t3','t5','t6' for 'Azure2017' branch;

## Run Basic Module
```
python3 SpotBlock-Main.py <data_branch> <typeid> <downSampleFactor> 
    <prediction_model> <whether optnet> <optnet violation regularity> <optnet iterations>
```
```
Example:  
```
- downSampleFactor: 0(no data) - 1(all data)
- prediction_model: one of 'LinearFit', 'TSDec', 'FCNet', 'LstmNet', 'AutoARIMA', 'FbProphet', 'UCM'
- optimization model: 'MIP' from gurobipy package; 'Heuristic' implemented by using idea of Heuristic Search
- whether optnet: 1(do optnet) / 0(not do optnet). Note that only 'FCNet' supports OptNet option.

## Run Bayesian Optimization Module
If you want to run with Bayesian Optimization Module, you can enter:

```
python3 SpotBlock-Main-BayesOpt.py <data_branch> <typeid> <downSampleFactor> <prediction_model> <whether optnet> --p=<violation_threshold>
```
```
Example: python3 SpotBlock-Main-BayesOpt.py Default 64 0.01 FCNet 1 --p=0.3
```
It will print average utilities and average viorate before/after Bayesian Optimization on the hyperparameters. The default number of interation is 10, you can change it in ''BayesOpt/BO_main.py''.

## Run The Script To Reproduce Experiment Results
If you want to run all the experiments end-to-end, it is easy to dirsctly run those scripts to reproduce the experiment results.

- to precache the ground truth results

```
python3 run-jobs-cache.py
```

- to run the experiments using basic module

```
python3 run-jobs-t4.py
```

- to run the experiments using bayesian optimization module

```
python3 run-jobs-t4-bo.py
```


## Experiment Results
Runs and saves the following results to an excel file with the corresponding filename. All the experiment results will be saved in the file 'exp_results/'.

- utilities & utility ratios & violation rates of two-stage model
- utilities & utility ratios & violation rates of RobustOpt model, for each ```p=0.05,0.10,...,0.50```.
- utilities & utility ratios & violation rates of optnet model if ```<whether optnet>==1```.
