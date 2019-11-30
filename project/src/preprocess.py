#!usr/bin/env Python
import numpy as np

def median_reference(x: np.ndarray):
    medians = np.median(x, axis=0)
    return x / medians

def normalize(x: np.ndarray):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x-means) / stds

def moving_average(x: np.ndarray, w: int):
    return np.convolve(x, np.ones(w), 'valid') / w

def snv(x: np.ndarray, axis=0: int):
    raise NotImplementedError
