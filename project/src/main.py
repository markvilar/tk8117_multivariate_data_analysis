import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from mpl_toolkits import mplot3d

from dataloader import Dataloader
from utilities import print_array_info, create_table, one_hot_encode
from plotting import data_inspection, pca_inspection

def data_calibration(dataloader):
    # Rawdata
    sdv = dataloader.get_samples().astype(float)
    labels = dataloader.get_labels()
    cali = dataloader.get_calibration()
    offset = cali['offset'][0,0]
    scale = cali['scale'][0,0]
    centers = np.squeeze(cali['centers'])
    # Calibration, change of units, non-negativity and radiance
    spec_rads = (sdv - offset) / scale
    spec_rads /= (0.01)**(-2)
    spec_rads[spec_rads<0] = 0
    rads = spec_rads * centers
    X = create_table(rads)
    Y = create_table(labels)
    return X, Y

def main():
    # Load data
    dir_path = '../datasets/classification/indian_pines'
    data_file = 'indian_pines.mat'
    cali_file = 'calibration.mat'
    labels_file = 'indian_pines_gt.mat'
    dataloader = Dataloader(dir_path, data_file, cali_file, labels_file)

    # Calibrate data and create tables
    X, Y = data_calibration(dataloader)

    # Select subset(?)

    # Create training and test set

    # PCA analysis

if __name__ == '__main__':
    main()
