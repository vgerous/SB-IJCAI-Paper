params_data_branch = [
    # 'Default',
    # 'Azure2017',
    'Azure2019'
]

params_data_typeid = [
    'm1', 
    'm2', 
    'm3', 
    # 'm4',
]
params_downsample_factor = ['1']
params_prediction_model = [
    'LinearFit',
    'TSDec',
    'AutoARIMA',
    'FbProphet',
    'UCM',
    'LstmNet',
    # 'FCNet',
]
params_sim_data_size = [
    'large',
    'middle',
    'small',
]
params_sim_data_index = [
    '33', '50', '10'
]

class Task:
    def __init__(self, *params):
        self.params = params
        self.index = 0
        self.id = '_'.join(params)
    def __str__(self):
        return str(self.index) + ': ' + self.id

    def get_command(self):
        b, t, d, m, s, i = self.params
        return [
            'python3',
            'SpotBlock-Main.py',
            b,
            t,
            d,
            m,
            '0',
            '--use-sim=1',
            f'--use-sim-size={s}',
            f'--use-sim-index={i}'
        ]

def get_all_tasks():
    task_list = list([
        Task(b, t, d, m, s, i)
        for b in params_data_branch
        for t in params_data_typeid
        for d in params_downsample_factor
        for m in params_prediction_model
        for s in params_sim_data_size
        for i in params_sim_data_index
    ])
    
    task_list.sort(key=lambda t: t.id)
    for ind, t in enumerate(task_list): t.index = ind
    return task_list

def run_tasks(tasks, proc=1):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    from os import path
    import datetime

    import subprocess

    def run_task(task):
        out_file = f'exp_results/job-output-t1.{task.id}.out'
        mark_file = f'exp_results/job-mark-t1.{task.id}.mark'
        print(f'[{task.index}] Executing task: {task} -> {" ".join(task.get_command())}')

        if path.exists(mark_file):
            print(f'[{task.index}] Skipping task {task}')
            return 0

        try:
            task_start_time = datetime.datetime.utcnow()
            with open(out_file, 'w') as stdout_file:
                result = subprocess.run(task.get_command(), stdout=stdout_file, stderr=stdout_file)
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

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    all_tasks = get_all_tasks()

    remaining_tasks = list(filter(lambda t: t.index % args.all_part == args.cur_part, all_tasks))

    run_tasks(remaining_tasks, proc=args.proc)
