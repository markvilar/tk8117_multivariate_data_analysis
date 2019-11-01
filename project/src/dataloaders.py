#!usr/bin/env Python
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict

from utilities import remove_dict_keys, print_dict_entries, print_array_info, one_hot_encode

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
        assert os.path.exists(dir_path), 'Folder does not exist: {}'.format(os.path.abspath(dir_path))
        assert os.path.isfile(data_path), 'File does not exist: {}'.format(os.path.abspath(data_path))
        assert os.path.isfile(cali_path), 'File does not exist: {}'.format(os.path.abspath(cali_path))
        assert os.path.isfile(labels_path), 'File does not exist: {}'.format(os.path.abspath(labels_path))
        paths = {'data': data_path, 'calibration': cali_path, 'labels': labels_path}
        self._paths = paths
        # Load files
        mat_headers = ['__header__', '__version__', '__globals__']
        data = remove_dict_keys(scipy.io.loadmat(data_path), mat_headers)['data']
        cali = remove_dict_keys(scipy.io.loadmat(cali_path), mat_headers)
        labels = remove_dict_keys(scipy.io.loadmat(labels_path), mat_headers)['labels']
        self._rawdata = {'data': data, 'calibration': cali, 'labels': labels}
        self._processed_data = None

    def create_test_set(self, test_frac: float):
        ''' Partitions the data into a test set and a training set. 
        arg test_frac: float, the fraction of the data used for testing '''
        assert test_frac > 0 and test_frac < 1, 'Fraction of test data must be between 0 and 1.'
        data = self._rawdata['data']
        labels = self._rawdata['labels']
        
        encoded_labels = one_hot_encode(labels)
        raise NotImplementedError

    def create_validation_folds(self, k: int):
        ''' Partitions the training set in k folds in order to do cross validation. 
        arg k: int, the number of folds '''
        raise NotImplementedError

def example_IP_dataloader():
    dataloader = IndianPinesDataloader('../datasets/classification/indian_pines', 
            'indian_pines.mat', 'calibration.mat', 'indian_pines_gt.mat')
    print_dict_entries(dataloader._rawdata)
    print_array_info(dataloader._rawdata['data'])

if __name__ == '__main__':
    example_IP_dataloader()
