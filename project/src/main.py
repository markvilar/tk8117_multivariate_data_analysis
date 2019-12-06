import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from dataloader import Dataloader
from plotting import data_inspection, pca_inspection, pls_inspection, kernel_pca_inspection
from preprocess import create_table, create_test_set, data_preprocess, remove_class
from preprocess import resample_dataset, hotellings_t2, inspect_outliers, normalize
from preprocess import remove_outliers
from classification import linear_classification, svm_cross_validation, svm_classification

def main():
    # Set seed for reproducability
    np.random.seed(6969)

    # Preprocess
    do_smoothing = False
    do_subset = False
    do_snv = False
    do_normalize = False
    
    # Analysis
    inspection = 'processed'
    show_plots = True
    do_outlier_filtering = False
    do_linear_class = False
    do_nonlinear_class = False

    # Load data
    dir_path = '../datasets/indian_pines'
    data_file = 'indian_pines.mat'
    cali_file = 'calibration.mat'
    labels_file = 'indian_pines_gt.mat'
    dataloader = Dataloader(dir_path, data_file, cali_file, labels_file)

    # Create tables, resample and create test set
    X = dataloader.get_calibrated_samples()
    Y = dataloader.get_labels()
    W = dataloader.get_wave_lengths()
    X = create_table(X)
    Y = create_table(Y)
    X, Y = resample_dataset(X, Y, 4.0)
    X_train, Y_train, X_test, Y_test = create_test_set(X, Y, test_frac=0.30)

    # Smoothing, subset selection, SNV
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
            do_smoothing, do_subset, do_snv, False)
    #W = dataloader.get_wave_lengths(subset_inds)

    # Outlier detection
    if do_outlier_filtering:
        outliers = hotellings_t2(X_train, Y_train, 0.05, True, False,
                fig_num=3, fig_size=(12,6))
        outliers = np.take(outliers, [2]) # 95% CI: Total 45; 2
        inspect_outliers(W, X_train, outliers)
        X_train, Y_train = remove_outliers(X_train, Y_train, outliers)
    
    # Normalization
    if do_normalize:
        X_train = normalize(X_train)
        X_test = normalize(X_test)

    # PCA inspection
    if inspection == 'processed':
        data_inspection(W, X_train, Y_train, 17, 1, (12,6), 'Raw Data', 
                'Wave lengths [nm]', 'Radiance [Wm^(-2)sr^(-1)]')
    elif inspection == 'pls':
        pls_inspection(X_train, Y_train, n_comps=8)
    elif inspection == 'pca':
        pca_inspection(X_train, Y_train, n_comps=8)
    elif inspection == 'kpca':
        kernel_pca_inspection(X_train, Y_train, 8, 'linear')

    # Linear classification
    if do_linear_class:
        linear_classification(X_train, Y_train, X_test, Y_test, n_folds=5, n_comps_max=10,
                threshold=0.90, show_plots=show_plots, fignum=2, figsize=(8,6), normalize=False)

    # Non-linear classification
    if do_nonlinear_class:
        gamma_min = 1e-5 #1e-5
        gamma_max = 3e-1 #3e-1
        n_gammas = 30 #30
        gammas = np.linspace(gamma_min, gamma_max, n_gammas)
        best_gamma = svm_cross_validation(X_train, Y_train, n_folds=5, kernel='rbf', 
                gammas=gammas, show_plots=show_plots)
        svm_classification(X_train, Y_train, X_test, Y_test, kernel='rbf', gamma=best_gamma)

if __name__ == '__main__':
    main()
