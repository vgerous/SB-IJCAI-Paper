import pandas as pd
import random
import json
import os

def generate_pms_forv2(ratio):
    print("generating_data")
    data = pd.read_csv("data/v2_p95_30_subset.csv")

    all_vm_config = {
        'core-2-|-memory-2':{
            "cpu_cores": 2,
            "memory_in_gb": 2,
        },
        'core-2-|-memory-4':{
            "cpu_cores": 2,
            "memory_in_gb": 4,
        },
        'core-2-|-memory-8':{
            "cpu_cores": 2,
            "memory_in_gb": 8,
        },
        'core-24-|-memory-64':{
            "cpu_cores": 24,
            "memory_in_gb": 64,
        },
        'core-4-|-memory-32':{
            "cpu_cores": 4,
            "memory_in_gb": 32,
        },
        'core-4-|-memory-8':{
            "cpu_cores": 4,
            "memory_in_gb": 8,
        },
        'core-8-|-memory-32':{
            "cpu_cores": 8,
            "memory_in_gb": 32,
        },
        'core-8-|-memory-64':{
            "cpu_cores": 8,
            "memory_in_gb": 64,
        },
        'core->24-|-memory->64':{
            "cpu_cores": 48,
            "memory_in_gb": 128,
        }
    }

    vm_types = data.columns.values
    resource_demand = {
        "cpu_cores": [],
        "memory_in_gb": [],
    }
    for i in range(179):
        cpu_demand = 0
        memory_demand = 0
        for vmt in vm_types:
            vmconfig = None
            for name, config in all_vm_config.items():
                if name in vmt:
                    vmconfig = config
                    break
            vm_demand = data.loc[i, vmt]
            cpu_demand += vmconfig["cpu_cores"] * vm_demand
            memory_demand += vmconfig["memory_in_gb"] * vm_demand
        resource_demand["cpu_cores"].append(cpu_demand)
        resource_demand["memory_in_gb"].append(memory_demand)
        print(i, cpu_demand, memory_demand)

    pm1 = [48,128]
    pm2 = [64,512]

    pm_list = []
    for i in range(179):
        random.seed(1)
        core_demand = resource_demand["cpu_cores"][i] 
        memory_demand = resource_demand["memory_in_gb"][i]            
        core_capacity = core_demand * 0.8                                                                                                                                                                   
        memory_capacity =memory_demand* ratio
        core_left, memory_left = core_capacity, memory_capacity

        original_ratio = memory_capacity / core_capacity
        pms = []
        while core_left > 0:
            if memory_left / core_left >= original_ratio:
                pm = pm2 
            else:
                pm = pm1
            pms.append(pm)
            core_left -= pm[0]
            memory_left -= pm[1]

        pm_list.append(pms)
        print(core_capacity, memory_capacity, sum([ele[0] for ele in pms]), sum([ele[1] for ele in pms]), original_ratio, sum([ele[1] for ele in pms])/sum([ele[0] for ele in pms]))

    pms_path = "data/pm_status_v2_p95_30_subset_pms_08core_%s"%str(ratio).replace(".", "")
    if not os.path.exists(pms_path):
        os.mkdir(pms_path)

        for i in range(1,179):
            pm_status = {}
            for index, pm in enumerate(pm_list[i]):
                pm_status[str(index)]={
                    "total": {
                        "cpu_cores": 96,
                        "memory_in_gb": 1024,
                    },
                    "free": {
                        "cpu_cores": pm[0],
                        "memory_in_gb": pm[1],
                    },
                    "used": {
                        "cpu_cores": 96-pm[0],
                        "memory_in_gb": 1024-pm[1],
                    },
                    "minimum_requirements": {
                        "cpu_cores": 2,
                        "memory_in_gb": 2,
                    }
                }
            with open(os.path.join(pms_path, "%d.json" % (i-1)),"w") as f:
                json.dump(pm_status, f)
        print("generating done")
    else:
        print("data already exists")
    return pms_path

def generate_pms_forv2_pure_core(ratio):
    print("generating_data")
    data = pd.read_csv("data/v2_p95_30_subset.csv")

    all_vm_config = {
        'core-2-|-memory-2':{
            "cpu_cores": 2,
            "memory_in_gb": 2,
        },
        'core-2-|-memory-4':{
            "cpu_cores": 2,
            "memory_in_gb": 4,
        },
        'core-2-|-memory-8':{
            "cpu_cores": 2,
            "memory_in_gb": 8,
        },
        'core-24-|-memory-64':{
            "cpu_cores": 24,
            "memory_in_gb": 64,
        },
        'core-4-|-memory-32':{
            "cpu_cores": 4,
            "memory_in_gb": 32,
        },
        'core-4-|-memory-8':{
            "cpu_cores": 4,
            "memory_in_gb": 8,
        },
        'core-8-|-memory-32':{
            "cpu_cores": 8,
            "memory_in_gb": 32,
        },
        'core-8-|-memory-64':{
            "cpu_cores": 8,
            "memory_in_gb": 64,
        },
        'core->24-|-memory->64':{
            "cpu_cores": 48,
            "memory_in_gb": 128,
        }
    }

    vm_types = data.columns.values
    resource_demand = {
        "cpu_cores": [],
        "memory_in_gb": [],
    }
    for i in range(179):
        cpu_demand = 0
        memory_demand = 0
        for vmt in vm_types:
            vmconfig = None
            for name, config in all_vm_config.items():
                if name in vmt:
                    vmconfig = config
                    break
            vm_demand = data.loc[i, vmt]
            cpu_demand += vmconfig["cpu_cores"] * vm_demand
            memory_demand += vmconfig["memory_in_gb"] * vm_demand
        resource_demand["cpu_cores"].append(cpu_demand)
        resource_demand["memory_in_gb"].append(memory_demand)
        print(i, cpu_demand, memory_demand)

    pm_list = []
    for i in range(179):
        core_demand = resource_demand["cpu_cores"][i] 
        core_capacity = core_demand * ratio
        pms = { 
            "1":
                {
                "total": {
                    "cpu_cores": core_capacity,
                },
                "free": {
                    "cpu_cores": core_capacity,
                },
                "used": {
                    "cpu_cores": 0,
                },
                "minimum_requirements": {
                    "cpu_cores": 2,
                }
            }
        }
        pm_list.append(pms)

    pms_path = "data/pm_status_v2_p95_30_subset_pms_pure_core_%s"%str(ratio).replace(".", "")
    if not os.path.exists(pms_path):
        os.mkdir(pms_path)
        for i, pm_status in enumerate(pm_list):
            if i == 0:
                continue
            with open(os.path.join(pms_path, "%d.json" % (i-1)),"w") as f:
                json.dump(pm_status, f)
        print("generating done")
    else:
        print("data already exists")
    return pms_path

def generate_pms_forv1_pure_core(ratio):
    print("generating_data")
    data = pd.read_csv("data/v1_avgcpu_30_2hour_1p2_subset_decycle.csv")

    all_vm_config = {
        'core-1-|-memory-0.75':{
            "cpu_cores":1,
            "memory_in_mb": 768,
        },
        'core-1-|-memory-1.75':{
            "cpu_cores":1,
            "memory_in_mb": 1792,
        },
        'core-2-|-memory-3.5':{
            "cpu_cores":2,
            "memory_in_mb": 3584,
        },
        'core-4-|-memory-7.0':{
            "cpu_cores":4,
            "memory_in_mb": 7168,
        },
        'core-8-|-memory-14.0':{
            "cpu_cores":8,
            "memory_in_mb": 14336,
        },
        'core-8-|-memory-56.0':{
            "cpu_cores":8,
            "memory_in_mb": 57344,
        },
    }

    vm_types = data.columns.values
    resource_demand = {
        "cpu_cores": [],
        "memory_in_mb": [],
    }
    for i in range(257):
        cpu_demand = 0
        memory_demand = 0
        for vmt in vm_types:
            vmconfig = None
            for name, config in all_vm_config.items():
                if name in vmt:
                    vmconfig = config
                    break
            vm_demand = data.loc[i, vmt]
            cpu_demand += vmconfig["cpu_cores"] * vm_demand
            memory_demand += vmconfig["memory_in_mb"] * vm_demand
        resource_demand["cpu_cores"].append(cpu_demand)
        resource_demand["memory_in_mb"].append(memory_demand)
        print(i, cpu_demand, memory_demand)

    pm_list = []
    for i in range(257):
        core_demand = resource_demand["cpu_cores"][i] 
        core_capacity = core_demand * ratio
        pms = { 
            "1":
                {
                "total": {
                    "cpu_cores": core_capacity,
                },
                "free": {
                    "cpu_cores": core_capacity,
                },
                "used": {
                    "cpu_cores": 0,
                },
                "minimum_requirements": {
                    "cpu_cores": 1,
                }
            }
        }
        pm_list.append(pms)

    pms_path = "data/pm_status_v1_avgcpu_30_2hour_1p2_subset_decycle_pure_core_%s"%str(ratio).replace(".", "")
    if not os.path.exists(pms_path):
        os.mkdir(pms_path)
        for i, pm_status in enumerate(pm_list):
            if i == 0:
                continue
            with open(os.path.join(pms_path, "%d.json" % (i-1)),"w") as f:
                json.dump(pm_status, f)
        print("generating done")
    else:
        print("data already exists")
    return pms_path