params_data_branch_type = [
    # Branch, Data, UseSim, SimSize, SimIndex
    # ('Default', 'm1', '0', '0', '0'),
    ('Azure2017', 'm2', '0', '0', '0'),
    # ('Azure2019', 'm3', '0', '0', '0'),
]

params_downsample_factor = [
    '0.01',
    '0.1',
    # '1'
]

params_prediction_model = [
    # Model, isOptNet, vioReg, optIter
    # ('LinearFit', '0', '0', '0'),
    # ('TSDec', '0', '0', '0'),
    # ('AutoARIMA', '0', '0', '0'),
    # ('FbProphet', '0', '0', '0'),
    # ('UCM', '0', '0', '0'),

    ('LstmNet', '0', '0', '0'),
    ('FCNet', '1', '50', '10'),
    ('LstmNet', '1', '50', '10'),
]

params_opt_time = [
    '30',
    # '600',
    '3600',
]

class Task:
    def __init__(self, *params):
        self.params = params
        self.index = 0
        self.id = '_'.join(params)
    def __str__(self):
        return str(self.index) + ': ' + self.id

    def get_command(self, job_proc=1):
        b, t, useSim, simSize, simIndex, d, m, isOp, vioReg, optIter, ot = self.params
        return [
            'python3',
            'SpotBlock-Main.py',
            b,
            t,
            d,
            m,
            *(
                (
                    isOp,
                    vioReg,
                    optIter,
                ) if isOp == '1' else ('0')
            ),
            *(
                (
                    f'--use-sim={useSim}',
                    f'--use-sim-size={simSize}',
                    f'--use-sim-index={simIndex}',
                ) if useSim == '1' else ()
            ),
            *(
                (
                    '--parallel=1',
                    f'--proc={job_proc}'
                ) if job_proc > 1 else ()
            ),
            f'--opt-time={ot}'
        ]

def get_all_tasks():
    task_list = list([
        Task(b, t, useSim, simSize, simIndex, d, m, isOp, vioReg, optIter, ot)
        for b, t, useSim, simSize, simIndex in params_data_branch_type
        for d in params_downsample_factor
        for m, isOp, vioReg, optIter in params_prediction_model
        for ot in params_opt_time
    ])
    
    task_list.sort(key=lambda t: t.id)
    for ind, t in enumerate(task_list): t.index = ind
    return task_list

def run_tasks(tasks, proc=1, job_proc=1):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from os import path
    import datetime

    import subprocess

    def run_task(task):
        out_file = f'exp_results/job-output-t4.{task.id}.out'
        mark_file = f'exp_results/job-mark-t4.{task.id}.mark'
        print(f'[{task.index}] Executing task: {task} -> {" ".join(task.get_command(job_proc=job_proc))}')

        if path.exists(mark_file):
            print(f'[{task.index}] Skipping task {task}')
            return 0

        try:
            task_start_time = datetime.datetime.utcnow()
            with open(out_file, 'w') as stdout_file:
                result = subprocess.run(task.get_command(job_proc=job_proc), stdout=stdout_file, stderr=stdout_file)
            task_end_time = datetime.datetime.utcnow()
            print(f'[{task.index}] Task finished {task} with return code {result.returncode} @ {task_end_time} taking {task_end_time - task_start_time}')

            if result.returncode == 0:
                with open(mark_file, 'w') as mark_f:
                    mark_f.write('')
            else:
                print(f'[{task.index}] Failed with task {task}')
        except Exception as e:
            print(e)
            return -1

    with ThreadPoolExecutor(max_workers=proc) as executor:
        results = list(tqdm(
            executor.map(run_task, tasks),
            total=len(tasks),
        ))

    return results

def get_args():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-part', type=int, default=1, help='Total partitions (machines) for jobs')
    parser.add_argument('--cur-part', type=int, default=0, help='Current partition (machine) index')
    parser.add_argument('--proc', type=int, default=6, help='Number of processes')
    parser.add_argument('--skip-mark', type=bool, default=True, help='Skip finished jobs by markers')
    parser.add_argument('--job-proc', type=int, default=1, help='Run each job in parallel with given parallism')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    all_tasks = get_all_tasks()

    remaining_tasks = list(filter(lambda t: t.index % args.all_part == args.cur_part, all_tasks))

    run_tasks(remaining_tasks, proc=args.proc, job_proc=args.job_proc)
