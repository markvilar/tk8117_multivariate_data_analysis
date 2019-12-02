import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from dataloader import Dataloader
from plotting import data_inspection, pca_inspection, pls_inspection, kernel_pca_inspection
from preprocess import create_tables, data_preprocess, remove_class
from classifiers import lda, svm

def linear_classification(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
        Y_test: np.ndarray):
    # Create k-folds
    # PCA - CV
    # LDA - CV
    # LDA - test
    #lda(X_train, Y_train, X_test, Y_test)
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

    # Create training and test set
    X = dataloader.get_calibrated_samples()
    Y = dataloader.get_labels()
    X_train, Y_train, X_test, Y_test = create_tables(X, Y, test_frac=0.30)

    # Data inspection
    #W = dataloader.get_wave_lengths()
    #data_inspection(W, X_train[0:500,:], Y_train[0:500], 17, 1, (8,6), 
            #'Radiance Spectra Samples', 'Wave Length [nm]', 'Radiance [Wm^(-2)sr^(-1)]')
    
    # Remove class
    X_train, Y_train = remove_class(X_train, Y_train, 0)

    # Data preprocess
    avg_window = 5
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
    subset_inds = np.concatenate((subset1, subset2, subset3, subset4,
        subset5, subset6, subset7, subset8, subset9))
    X_train, X_test = data_preprocess(X_train, X_test, avg_window, subset_inds,
            do_smoothing=True, do_subset=True, do_snv=True, do_normalize=True)
    W = dataloader.get_wave_lengths(subset_inds)

    # PCA inspection
    pls_inspection(X_train, Y_train, n_comps=8)
    #pca_inspection(X_train, Y_train)
    #kernel_pca_inspection(X_train, Y_train, 8, 'cosine')
    # Linear classification
    linear_classification(X_train, Y_train, X_test, Y_test)

    # Non-linear classification

if __name__ == '__main__':
    main()
