'''
Author Junbong Jang
Date 7/14/2020

Contains helper functions used in UserParams.py to get parameters and GPUs
'''

import re
import subprocess as sp


def find_param_after_string(string):
    regex = re.compile(f'{string}([0-9]*)')
    print(regex.findall(string))


def get_MTL_weights(strategy_type):
    cls = re.findall(r"cls([0-9.]*[0-9]+)_", strategy_type)
    reg = re.findall(r"reg([0-9.]*[0-9]+)_", strategy_type)
    aut = re.findall(r"aut([0-9.]*[0-9]+)_", strategy_type)
    seg = re.findall(r"_seg([0-9.]*[0-9]+)", strategy_type)
    
    def process_regex_find(regex_result):
        if len(regex_result):
            return float(regex_result[0])
        else:
            return 0
    cls, reg, aut, seg = process_regex_find(cls), process_regex_find(reg), process_regex_find(aut), process_regex_find(seg)

    print('cls, reg, aut, seg weights:', cls, reg, aut, seg)

    return cls, reg, aut, seg


def get_MTL_auto_remove_task(strategy_type):
    m = re.search("VGG19_MTL_auto_(\w+)", strategy_type)
    removed_tasks = m.groups()[0].split('_')

    print('removed tasks', removed_tasks)

    return removed_tasks


def get_available_gpu():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY_GB = 23
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]

    available_gpu = None
    for i, x in enumerate(memory_free_info):
        gpu_memory_free_value = int(x.split()[0]) / 1024
        print(f'{i} GPU available memory: {gpu_memory_free_value} GB')
        if gpu_memory_free_value > ACCEPTABLE_AVAILABLE_MEMORY_GB:
            available_gpu = f'{i}'
            break
    if available_gpu is None:
        raise ValueError('No available GPU')
    print('Available GPU:', available_gpu)
    return available_gpu