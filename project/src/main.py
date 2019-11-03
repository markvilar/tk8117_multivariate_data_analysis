import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

from dataloaders import IndianPinesDataloader
from utilities import print_array_info

def data_inspection(dataloader):
    # Rawdata
    X, Y = dataloader.get_tables()

    # PCA
    pca = PCA()
    scores = pca.fit_transform(X)
    loadings = pca.components_
    var_ratios = pca.explained_variance_ratio_
    cum_var_ratios = np.cumsum(var_ratios)

    # Explained variance plot
    plt.figure(num=0, figsize=(8,6))
    plt.plot(np.pad(cum_var_ratios, (1,0), 'constant'))
    plt.title('Explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative explained variance')
    plt.xlim((0, 10))
    plt.ylim((0, 1))

    # 2D scores plot
    plt.figure(num=1, figsize=(8,6))
    plt.scatter(scores[:,0], scores[:,1], s=1, c=Y)
    plt.title('Scores plot')
    plt.xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    plt.ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))

    # 3D scores plot
    fig = plt.figure(num=2, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=Y)
    ax.set_xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    ax.set_ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))
    ax.set_zlabel('PC3 ({:.2f}% explained variance)'.format(var_ratios[2]*100))

    plt.show()
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
