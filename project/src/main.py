import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from mpl_toolkits import mplot3d

from dataloader import Dataloader
from utilities import print_array_info, create_table, create_test_set
from plotting import plot_spectras, pca_inspection, create_colormap, kernel_pca_inspection
from preprocess import snv, subset_selection

def data_inspection(dataloader: Dataloader, plot: bool):
    # Calibrate data and create tables
    wave_lengths = np.squeeze(dataloader.get_calibration('centers'))
    X = dataloader.get_calibrated_samples()
    Y = dataloader.get_labels()
    X, Y = create_table(X), create_table(Y)

    # Create training and test set
    X_train, Y_train, X_test, Y_test = create_test_set(X, Y, frac=0.30)
    colors_train = create_colormap(Y_train)

    # Subset selection
    # 0-38, 42-44, 48-53, 65-73, 84-86, 91-98, 120-144, 167-169, 172-220
    subset1 = np.arange(0, 38)
    subset2 = np.arange(42, 44)
    subset3 = np.arange(48, 53)
    subset4 = np.arange(65, 73)
    subset5 = np.arange(84, 86)
    subset6 = np.arange(91, 98)
    subset7 = np.arange(120, 144)
    subset8 = np.arange(167, 169)
    subset9 = np.arange(172, 220)
    indices = np.concatenate((subset5, subset6, subset7, subset8, subset9))
    X_train = subset_selection(X_train, indices)
    X_test = subset_selection(X_test, indices)
    wave_lengths = subset_selection(wave_lengths[np.newaxis,:], indices)
    wave_lengths = np.squeeze(wave_lengths)

    # SNV
    X_train = snv(X_train)
    X_test = snv(X_test)

    # Data inspection
    if plot:
        plot_spectras(wave_lengths, X_train[0:500,:], colors_train[0:500,:], 1, (8,6), '', '', '')

    # PCA inspection
    if plot:
        pca_inspection(X_train, Y_train)

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
    X_train, Y_train, X_test, Y_test = data_inspection(dataloader, plot=False)

    # Linear classification
    

    # Non-linear classification

if __name__ == '__main__':
    main()
