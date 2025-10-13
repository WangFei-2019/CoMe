from typing import List, Optional
import os
import logging
import random
import copy

import numpy as np
import torch

def list_item_sum(lst1, lst2):
    return [i+j for i, j in zip(lst1, lst2)]

def L2_norm(matrix, axis=-1):
    return torch.linalg.norm(matrix.float(), axis=axis)

def mean(matrix, axis=-1):
    return torch.mean(matrix.float(), axis=axis)

def var(matrix, axis=-1):
    return torch.var(matrix.float(), correction=0, dim=axis)

def none(matrix, axis=None):
    return matrix

def normalize(arr, axis=None):
    min_val = torch.min(arr, dim=axis, keepdim=True).values
    max_val = torch.max(arr, dim=axis, keepdim=True).values
    if (max_val - min_val).le(1e-8).any():
        raise ZeroDivisionError
    return (arr - min_val) / (max_val - min_val)

def softmax(arr, axis=None):
    exp_arr = torch.exp(arr - torch.max(arr, dim=axis, keepdim=True).values)
    sum_exp = torch.sum(exp_arr, dim=axis, keepdim=True)
    return exp_arr / sum_exp

METHODS= {"l2": L2_norm, "mean": mean, "var": var, "none": none, "norm": normalize, "softmax": softmax}


def get_nested_attribute(obj, attr):
    """通过字符串路径获取嵌套属性"""
    attributes = attr.split('.')
    for attribute in attributes:
        obj = getattr(obj, attribute)
    return obj

def distribute_and_round(integer, merge_raio):
    floored_values = []
    # Step 1: Multiply and floor
    max_idx = merge_raio.index(max(merge_raio))
    for i in range(len(merge_raio)):
        if i != max_idx:
            floored_values.append(int(integer * merge_raio[i]))
        else:
            floored_values.append(0)
    floored_values[max_idx] = integer - sum(floored_values)

    return floored_values

def concat_list(l1:List[torch.tensor], l2:List[torch.tensor], axis=0):
    assert len(l1)==len(l2)
    l3 = []
    for i in range(len(l1)):
        l3.append(torch.cat((l1[i], l2[i]), dim=axis))
    return l3

def are_tensors_on_same_device(tensor_list):
    if not tensor_list:
        return True 
    first_device = tensor_list[0].device
    return all(tensor.device == first_device for tensor in tensor_list)

def leaf_node_generate(layer_info:list, merge_l:list):
    merge_list = copy.copy(merge_l)
    if not all(isinstance(element, list) for element in layer_info):
        layer_info = [layer_info]
    list_before_merge = copy.copy(layer_info[-1])
    merge_list.sort(reverse=True)

    merge_layers = []
    for i in merge_list:
        merge_layers.append(list_before_merge[i])
        del list_before_merge[i]
    
    merge_layers = sorted(merge_layers, key=get_first_non_list_element)

    list_before_merge.insert(merge_list[-1], merge_layers)
    layer_info.append(list_before_merge)
    return layer_info

def get_first_non_list_element(item):
    while isinstance(item, list):
        item = item[0]
    return item

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def set_seed(seed: int = 1):
    """
    Sets the random seed for reproducibility across various libraries and environments.

    Args:
        seed (int, optional): The seed value to set. Defaults to 1.
    """
    random.seed(seed)  # Set seed for Python's random module
    np.random.seed(seed)  # Set seed for NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set seed for Python hash-based operations
    torch.manual_seed(seed)  # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.benchmark = False     # Disable cuDNN benchmark for consistency

def init_logging(log_file: str = None, log_level: int = logging.INFO):
    """
    Initialize logging configuration.

    Args:
        log_file (str): Path to the log file. If None, logs will not be saved to a file.
        log_level (int): Logging level, default is logging.INFO.
    """
    # Define log format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure log handlers
    handlers = [logging.StreamHandler()]  # Default to console output

    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Add file handler for logging to a file
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Test log
    logging.info("Logging system initialized.")