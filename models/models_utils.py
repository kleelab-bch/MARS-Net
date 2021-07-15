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