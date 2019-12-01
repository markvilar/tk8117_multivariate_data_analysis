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

def snv(x: np.ndarray):
    assert x.ndim == 2, 'x must be a 2D array.'
    means = np.mean(x, axis=1)
    stds = np.std(x, axis=1)
    return (x - means[:, np.newaxis]) / stds[:, np.newaxis]

def subset_selection(x: np.ndarray, indices: np.ndarray):
    n_features = x.shape[1]
    mask = np.logical_and(indices < n_features, indices >= 0)
    indices = indices[mask]
    indices = np.unique(indices)
    indices = np.sort(indices)
    return np.take(x, indices, axis=1)
