import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition

from typing import Tuple

def plot2D(x: np.ndarray, y: np.ndarray, c: np.ndarray, num: int, size: Tuple[int, int]):
    plt.figure(num=num, figsize=size)
    plt.plot(x.T, y.T)
    plt.show()

def plot_spectras(wave_lengths: np.ndarray, samples: np.ndarray, labels: np.ndarray, num: int, size: Tuple[int, int]):
    wave_lengths = np.tile(wave_lengths, (samples.shape[0], 1))
    plot2D(wave_lengths, samples, labels, num, size)

def pca_inspection(dataloader, plot: bool):
    rads = dataloader.get_rads()
    labels = dataloader.get_labels()
    X = create_table(rads)
    labels = create_table(labels)
    n_classes = len(np.unique(labels))

    # PLS
    pca = decomposition.PCA()
    scores = pca.fit_transform(X)
    loadings = pca.components_
    var_ratios = pca.explained_variance_ratio_
    cum_var_ratios = np.cumsum(var_ratios)

    # Colormap
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom map',
            cmaplist, cmap.N)
    bounds = np.linspace(0, n_classes, n_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

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
        plt.figure(num=4, figsize=(8,6))
        plt.plot(loadings[0,:])
        plt.title('PC1 loadings')
        
        plt.figure(num=5, figsize=(8,6))
        plt.plot(loadings[1,:])
        plt.title('PC2 loadings')

        # 2D scores plot
        plt.figure(num=6, figsize=(8,6))
        scat = plt.scatter(scores[:,0], scores[:,1], c=labels, s=1,
                cmap=cmap, norm=norm)
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
        cb.set_label('Classes')
        plt.title('Scores plot')
        plt.xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
        plt.ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))

        # 3D scores plot
        fig = plt.figure(num=7, figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=labels, s=1,
                cmap=cmap, norm=norm)
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
        cb.set_label('Classes')
        ax.set_title('Scores plot')
        ax.set_xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
        ax.set_ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))
        ax.set_zlabel('PC3 ({:.2f}% explained variance)'.format(var_ratios[2]*100))

        plt.show()
