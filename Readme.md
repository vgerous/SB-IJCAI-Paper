# Note
## Dependencies
- **python3**, **numpy**, **torch**, **cvxpy**, **gurobi**, **scikit-learn==0.22.2.post1**
- any version not too old should work.

## How to run
```
python3 SpotBlock-Main.py <data_branch> <typeid> <downSampleFactor> 
    <prediction_model> <whether optnet> <optnet violation regularity> <optnet iterations>
Example:  
```
For precaching ground truth results, run:
```
python3 SpotBlock-Run-GroundTruth.py <data_branch> <typeid> <downSampleFactor>
Example: python3 SpotBlock-Run-GroundTruth.py Azure2019 t1 0.01
```

- data_branch: 'Default', 'Azure2019', 'Azure2017'
- typeid: 65, 64, 42, 41 for 'Default' data branch; 't1','t2','t3','t5','t6','t7' for 'Azure2019' branch; 't1','t2','t3','t5','t6' for 'Azure2017' branch;
- downSampleFactor: 0(no data) - 1(all data)
- prediction_model: one of 'LinearFit', 'TSDec', 'FCNet', 'LstmNet', 'AutoARIMA', 'FbProphet', 'UCM'
- whether optnet: 1(do optnet) / 0(not do optnet). Note that only 'FCNet' supports OptNet option.
- If you want to run with Bayesian Optimization Module, you can enter:

```
python3 SpotBlock-Main-BayesOpt.py <data_branch> <typeid> <downSampleFactor> <prediction_model> <whether optnet> --p=<violation_threshold>
Example: python3 SpotBlock-Main-BayesOpt.py Default 64 0.01 FCNet 1 --p=0.3
```
- It will print average utilities and average viorate before/after Bayesian Optimization on the hyperparameters. The default #interation = 10, you can change it in ''BayesOpt/BO_main.py''.

## What it does
Runs and saves the following results to an excel file with the corresponding filename.
- utilities & utility ratios & violation rates of two-stage model
- utilities & utility ratios & violation rates of RobustOpt model, for each ```p=0.05,0.10,...,0.50```.
- utilities & utility ratios & violation rates of optnet model if ```<whether optnet>==1```.
## Others
- there's a ```default_capacity_scaledown``` parameter for each branch & type of data (see the main file for details). 
Feel free to change them to make the problem instances more/less challenging.
- Sometimes Gurobi gets stuck when solving MIP. Currently a maximum time limit of 30 seconds is set; feel free to change it.
- Sometimes the qpth solver provided by the OptNet paper breaks down; nothing can be done except re-run the codes.