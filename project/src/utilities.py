#!/usr/bin/env Python
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple
from queue import Queue

def remove_dict_keys(data: Dict, to_remove: List):
    ''' Remove the dictionary entries based on the entry key. '''
    for key in to_remove:
        del data[key]
    return data

def remove_array_entries(arr, indices):
    ''' Remove the array entries with the given indices.
    arg arr: Nx1 array of floats
    arg indices: list of ints '''
    n = arr.shape[0]
    mask = np.full(arr.shape, True, dtype=bool)
    for i in indices:
        if i >= n or i < 0:
            continue
        else:
            mask[i,:] = False

    return arr[mask]

def print_dict_entries(x: Dict):
    ''' Prints the key and value type for every dictionary entry. '''
    for key, value in x.items():
        print('key: {}, value: {}'.format(key, type(value)))

def print_array_info(x: np.ndarray):
    ''' Prints the shape and data type of a numpy array. '''
    print('shape: {}, dtype: {}'.format(x.shape, x.dtype))

def create_table(x: np.ndarray) -> np.ndarray:
    ''' Reshapes an ND array to 2D.
    arg x: ND array
    return; 2D array '''
    if x.ndim <= 2:
        return x.reshape(-1)
    return x.reshape(-1, x.shape[-1])

def create_test_set(X: np.ndarray, Y: np.ndarray, frac: float):
    
    raise NotImplementedError
    return X_train, Y_train, X_test, Y_test

def one_hot_encode(labels: np.ndarray) -> Tuple[Dict, np.ndarray]:
    ''' One hot encodes labels. 
    arg labels: WxHxP np.array of uints
    return: tuple of dict and np.ndarray '''
    n_samples = len(labels)
    classes = np.unique(labels)
    n_classes = len(classes)
    if n_classes < 256:
        data_type = np.uint8
    else:
        data_type = np.uint16
    encoded_labels = np.zeros((n_samples, n_classes), dtype=data_type)
    encoded_labels[np.arange(n_samples), labels] = 1
    encoded_labels = encoded_labels.reshape((n_samples, n_classes))
    return encoded_labels


