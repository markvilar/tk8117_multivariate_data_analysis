import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dataloaders import IndianPinesDataloader
from utilities import print_array_info

def data_inspection(dataloader):
    # plot rawdata
    X, Y = dataloader.get_tables()

    # perform pca
    pca = PCA()
    scores = pca.fit_transform(X)
    loadings = pca.components_
    var_ratios = pca.explained_variance_ratio_
    cum_var_ratios = np.cumsum(var_ratios)

    print_array_info(scores)
    print_array_info(loadings)
    print_array_info(var_ratios)
    print_array_info(cum_var_ratios)

    plt.figure(num=0, figsize=(6,6))
    plt.plot(cum_var_ratios)
    plt.xlim((0, 10))
    plt.ylim((0, 1))
    plt.show()

    # plot pca scores in 2D

    # plot pca scores in 3D
    raise NotImplementedError

def data_preprocess(dataloader):
    raise NotImplementedError 

def lda_classification(dataloader):
    raise NotImplementedError

def svm_classification(dataloader):
    raise NotImplementedError

def main():
    dir_path = '../datasets/classification/indian_pines'
    data_file = 'indian_pines_corrected.mat'
    cali_file = 'calibration_corrected.mat'
    labels_file = 'indian_pines_gt.mat'
    dataloader = IndianPinesDataloader(dir_path, data_file, cali_file, labels_file)
    data_inspection(dataloader)

if __name__ == '__main__':
    main()
