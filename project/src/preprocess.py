#!usr/bin/env Python
import numpy as np

def median_reference(x: np.ndarray):
    medians = np.median(x, axis=0)
    return x / medians

def normalize(x: np.ndarray):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x-means) / stds
