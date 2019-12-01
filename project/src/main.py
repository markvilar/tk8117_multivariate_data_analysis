import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from mpl_toolkits import mplot3d

from dataloader import Dataloader
from utilities import print_array_info, create_table, create_test_set
from plotting import plot_spectras, pca_inspection

def data_inspection(dataloader: Dataloader):
    # Calibrate data and create tables
    wave_lengths = np.squeeze(dataloader.get_calibration('centers'))
    X = dataloader.get_calibrated_samples()
    Y = dataloader.get_labels()
    X, Y = create_table(X), create_table(Y)

    # Select subset(?)

    # Create training and test set
    X_train, Y_train, X_test, Y_test = create_test_set(X, Y, frac=0.30)

    # Data inspection
    plot_spectras(wave_lengths, X_train[0:200,:], Y_train[0:200], 1, (8, 6))

    # PCA inspection

    # Outlier detection

    # Data inspection 2

    # PCA inspection

    return X_train, Y_train, X_test, Y_test

def linear_classification():
    raise NotImplementedError

def nonlinear_classification():
    raise NotImplementedError

def main():
    # Set seed for reproducability
    np.random.seed(6969)

    # Load data
    dir_path = '../datasets/indian_pines'
    data_file = 'indian_pines.mat'
    cali_file = 'calibration.mat'
    labels_file = 'indian_pines_gt.mat'
    dataloader = Dataloader(dir_path, data_file, cali_file, labels_file)
    
    # Data inspection, PCA inspection and outlier detection
    data_inspection(dataloader)

    # Linear classification

    # Non-linear classification

if __name__ == '__main__':
    main()
