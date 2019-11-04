#!usr/bin/env Python
import os
import scipy
import scipy.io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Dict, Tuple

from utilities import remove_dict_keys, create_table, one_hot_encode
from preprocess import median_reference, normalize

class IndianPinesDataloader():
    def __init__(self, dir_path: str, data_file: str, cali_file: str, labels_file: str):
        ''' Dataloader class for the Indian Pines hyperspectral image dataset.
        arg dir_path: path to dataset directory
        arg data_file: name of the data file
        arg cali_file: name of the calibration file
        arg labels_file: name of the labels file 
        return: IndianPinesDataloader object'''
        # File paths
        data_path = dir_path + '/' + data_file
        cali_path = dir_path + '/' + cali_file
        labels_path = dir_path + '/' + labels_file
        assert os.path.exists(dir_path), 'Folder does not exist: {}'.format(
                os.path.abspath(dir_path))
        assert os.path.isfile(data_path), 'File does not exist: {}'.format(
                os.path.abspath(data_path))
        assert os.path.isfile(cali_path), 'File does not exist: {}'.format(
                os.path.abspath(cali_path))
        assert os.path.isfile(labels_path), 'File does not exist: {}'.format(
                os.path.abspath(labels_path))
        paths = {'data': data_path, 'calibration': cali_path, 'labels': labels_path}
        self._paths = paths
        # Load files
        mat_headers = ['__header__', '__version__', '__globals__']
        samples = remove_dict_keys(scipy.io.loadmat(data_path), mat_headers)['data']
        cali = remove_dict_keys(scipy.io.loadmat(cali_path), mat_headers)
        labels = remove_dict_keys(scipy.io.loadmat(labels_path), mat_headers)['labels']
        self._samples = samples # scans, sensors, channels
        self._calibration = cali
        self._labels = labels
        self._spec_rads = None
        self._rads = None
        self._X = None
        self._Y = None

    def get_samples(self):
        return self._samples.copy()

    def set_rads(self, x: np.ndarray):
        self._rads = x

    def get_rads(self) -> np.ndarray:
        return self._rads.copy()

    def set_spec_rads(self, x: np.ndarray):
        self._spec_rads = x

    def get_spec_rads(self) -> np.ndarray:
        return self._spec_rads.copy()

    def get_calibration(self, key: str=None) -> Dict:
        if key in self._calibration:
            return self._calibration[key]
        elif key == None:
            return self._calibration
        else:
            return None

    def get_labels(self) -> np.ndarray:
        return self._labels.copy()

    def set_tables(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndims == 2, 'Wrong dimensionality. Dimensionality of X: {}\n'.format(X.ndims)
        assert Y.ndims == 2, 'Wrong dimensionality. Dimensionality of Y: {}\n'.format(Y.ndims)

    def get_tables(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._X.copy(), self._Y.copy()

    def create_test_set(self, test_frac: float):
        ''' Partitions the data into a test set and a training set. 
        arg test_frac: float, the fraction of the data used for testing '''
        assert test_frac > 0 and test_frac < 1, 'Fraction of test data must be between 0 and 1.'
        raise NotImplementedError

    def create_validation_folds(self, k: int):
        ''' Partitions the training set in k folds in order to do cross validation. 
        arg k: int, the number of folds '''
        raise NotImplementedError

def example_IP_dataloader():
    dataloader = IndianPinesDataloader('../datasets/classification/indian_pines', 
            'indian_pines_corrected.mat', 'calibration_corrected.mat', 'indian_pines_gt.mat')

if __name__ == '__main__':
    example_IP_dataloader()
