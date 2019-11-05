import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from mpl_toolkits import mplot3d

from dataloaders import IndianPinesDataloader
from utilities import print_array_info, create_table, one_hot_encode, moving_average

def data_calibration(dataloader):
    # Rawdata
    sdv = dataloader.get_samples().astype(float)
    cali = dataloader.get_calibration()
    offset = cali['offset'][0,0]
    scale = cali['scale'][0,0]
    centers = np.squeeze(cali['centers'])
    # Calibration, change of units, non-negativity and radiance
    spec_rads = (sdv - offset) / scale
    spec_rads /= (0.01)**(-2)
    spec_rads[spec_rads<0] = 0
    rads = spec_rads * centers
    # Set processed samples
    dataloader.set_spec_rads(spec_rads)
    dataloader.set_rads(rads)

def data_inspection(dataloader, plot: bool):
    spec_rads = dataloader.get_spec_rads()
    rads = dataloader.get_rads()
    centers = dataloader.get_calibration('centers')
    centers = np.squeeze(centers)
    # Extract fake RGB
    rgb_img = np.take(spec_rads, [19,15,4], axis=2)
    rgb_img *= (1 / np.max(rgb_img))
    # Plots
    if plot:
        # Fake RGB plot
        plt.figure(num=0, figsize=(6,6))
        plt.imshow(rgb_img)
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                left=False, right=False, labelbottom=False, labeltop=False,
                labelleft=False, labelright=False)
        # Spectral Radiance plot
        plt.figure(num=1, figsize=(8,6))
        plt.plot(centers, spec_rads[0,:,:].T)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Spectral Radiance [Wm^(-2)nm^(-1)sr^(-1)]')
        # Radiance plot
        plt.figure(num=2, figsize=(8,6))
        plt.plot(centers, rads[0,:,:].T)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Radiance [Wm^(-2)sr^(-1)]')
        plt.show()

def pca_inspection(dataloader, plot: bool):
    rads = dataloader.get_rads()
    labels = dataloader.get_labels()
    X = create_table(rads)
    labels = create_table(labels)
    Y = one_hot_encode(labels)
    # PLS
    pca = decomposition.PCA()
    x_scores = pca.fit_transform(X)
    loadings = pca.components_
    var_ratios = pca.explained_variance_ratio_
    cum_var_ratios = np.cumsum(var_ratios)
    print(loadings.shape)
    if plot:
        # Explained variance plot
        plt.figure(num=3, figsize=(8,6))
        plt.plot(np.pad(cum_var_ratios, (1,0), 'constant'))
        plt.title('Explained variance')
        plt.xlabel('Principal components')
        plt.ylabel('Cumulative explained variance')
        plt.xlim((0, 10))
        plt.ylim((0, 1))

        # Loadings plot
        plt.figure(num=3, figsize=(8,6))

        # 2D scores plot
        plt.figure(num=5, figsize=(8,6))
        plt.scatter(x_scores[:,0], x_scores[:,1], s=1, c=labels)
        plt.title('Scores plot')
        plt.xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
        plt.ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))

        # 3D scores plot
        fig = plt.figure(num=6, figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_scores[:,0], x_scores[:,1], x_scores[:,2], c=labels)
        ax.set_xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
        ax.set_ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))
        ax.set_zlabel('PC3 ({:.2f}% explained variance)'.format(var_ratios[2]*100))

        plt.show()

def data_preprocess(dataloader, plot: bool):
    rads = dataloader.get_rads()
    filtered_rads = moving_average(rads, 5)

def lda_classification(dataloader):
    raise NotImplementedError

def svm_classification(dataloader):
    raise NotImplementedError

def main():
    dir_path = '../datasets/classification/indian_pines'
    data_file = 'indian_pines.mat'
    cali_file = 'calibration.mat'
    labels_file = 'indian_pines_gt.mat'
    dataloader = IndianPinesDataloader(dir_path, data_file, cali_file, labels_file)
    data_calibration(dataloader)
    data_inspection(dataloader, False)
    data_preprocess(dataloader, False)
    pca_inspection(dataloader, False)

if __name__ == '__main__':
    main()
