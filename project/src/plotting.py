import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition

from typing import Tuple

from preprocess import one_hot_encode

def data_inspection(W: np.ndarray, X: np.ndarray, Y: np.ndarray, cnum: int, num: int, 
        size: Tuple[int, int], title: str, xlabel: str, ylabel:str):
    W = np.tile(W, (X.shape[0], 1))
    plt.figure(num=num, figsize=size)
    plt.plot(W.T, X.T)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def kernel_pca_inspection(X: np.ndarray, Y: np.ndarray, n_comps: int, kernel: str='linear'):
    n_classes = len(np.unique(Y))
    pca = decomposition.KernelPCA(n_components=n_comps, kernel=kernel)
    scores = pca.fit_transform(X)
    alphas = pca.alphas_
    eigen_vals = pca.lambdas_
    eigen_sum = np.sum(eigen_vals)
    eigen_ratios = eigen_vals / eigen_sum
    cum_eigen_ratios = np.cumsum(eigen_ratios)
    
    # Colormap
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom map',
            cmaplist, cmap.N)
    bounds = np.linspace(0, n_classes, n_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Explained variance plot
    plt.figure(num=3, figsize=(8,6))
    plt.plot(np.pad(cum_eigen_ratios, (1,0), 'constant'))
    plt.title('Eigenvalue Ratios')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative eigenvalue ratios')
    plt.xlim((0, n_comps))
    plt.ylim((0, 1))

    # 2D scores plot
    plt.figure(num=6, figsize=(8,6))
    scat = plt.scatter(scores[:,0], scores[:,1], c=Y, s=1,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    plt.title('Scores plot (KPCA, {} kernel)'.format(kernel))
    plt.xlabel('PC1 ({:.2f}% eigenvalue ratio)'.format(eigen_ratios[0]*100))
    plt.ylabel('PC2 ({:.2f}% eigenvalue ratio)'.format(eigen_ratios[1]*100))

    # 3D scores plot
    fig = plt.figure(num=7, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=Y, s=2,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax.set_title('Scores plot (KPCA, {} kernel)'.format(kernel))
    ax.set_xlabel('PC1 ({:.2f}% eigenvalue ratio)'.format(eigen_ratios[0]*100))
    ax.set_ylabel('PC2 ({:.2f}% eigenvalue ratio)'.format(eigen_ratios[1]*100))
    ax.set_zlabel('PC3 ({:.2f}% eigenvalue ratio)'.format(eigen_ratios[2]*100))

    plt.show()

def pca_inspection(X: np.ndarray, Y: np.ndarray, n_comps: int):
    n_classes = len(np.unique(Y))
    pca = decomposition.PCA(n_components=n_comps)
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

    # Explained variance plot
    plt.figure(num=3, figsize=(8,6))
    plt.plot(np.pad(cum_var_ratios, (1,0), 'constant'))
    plt.title('Explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative explained variance')
    plt.xlim((0, n_comps))
    plt.ylim((0, 1))

    # Loadings plot
    plt.figure(num=4, figsize=(8,6))
    plt.plot(loadings[0,:])
    plt.title('PC1 loadings')
    
    plt.figure(num=5, figsize=(8,6))
    plt.plot(loadings[1,:])
    plt.title('PC2 loadings')

    plt.figure(num=6, figsize=(8,6))
    plt.plot(loadings[2,:])
    plt.title('PC3 loadings')

    # 2D scores plot
    plt.figure(num=7, figsize=(8,6))
    scat = plt.scatter(scores[:,0], scores[:,1], c=Y, s=1,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    plt.title('Scores plot (PCA)')
    plt.xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    plt.ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))

    # 3D scores plot
    fig = plt.figure(num=8, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=Y, s=2,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax.set_title('Scores plot (PCA)')
    ax.set_xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    ax.set_ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))
    ax.set_zlabel('PC3 ({:.2f}% explained variance)'.format(var_ratios[2]*100))

    plt.show()

def pls_inspection(X: np.ndarray, Y: np.ndarray, n_comps: int):
    n_classes = len(np.unique(Y))
    Y_encoded = one_hot_encode(Y)
    model = cross_decomposition.PLSRegression(n_components=n_comps, scale=False)
    model.fit(X, Y_encoded)

    # Extract information
    scores = model.x_scores_
    loadings = model.x_loadings_
    var_scores = np.var(scores, axis=0)
    var_X = np.sum(np.var(X, axis=0))
    var_ratios = var_scores / var_X
    cum_var_ratios = np.cumsum(var_ratios)

    # Colormap
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom map',
            cmaplist, cmap.N)
    bounds = np.linspace(0, n_classes, n_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Explained variance plot
    plt.figure(num=3, figsize=(8,6))
    plt.plot(np.pad(cum_var_ratios, (1,0), 'constant'))
    plt.title('Explained variance')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative explained variance')
    plt.xlim((0, n_comps))
    plt.ylim((0, 1))

    # Loadings plot
    plt.figure(num=4, figsize=(8,6))
    plt.plot(loadings[:,0])
    plt.title('PC1 loadings')
    
    plt.figure(num=5, figsize=(8,6))
    plt.plot(loadings[:,1])
    plt.title('PC2 loadings')

    plt.figure(num=6, figsize=(8,6))
    plt.plot(loadings[:,2])
    plt.title('PC3 loadings')

    # 2D scores plot
    plt.figure(num=7, figsize=(8,6))
    scat = plt.scatter(scores[:,0], scores[:,1], c=Y, s=2,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    plt.title('Scores plot (PLS)')
    plt.xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    plt.ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))

    # 3D scores plot
    fig = plt.figure(num=8, figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scat = ax.scatter(scores[:,0], scores[:,1], scores[:,2], c=Y, s=2,
            cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Classes')
    ax.set_title('Scores plot (PLS)')
    ax.set_xlabel('PC1 ({:.2f}% explained variance)'.format(var_ratios[0]*100))
    ax.set_ylabel('PC2 ({:.2f}% explained variance)'.format(var_ratios[1]*100))
    ax.set_zlabel('PC3 ({:.2f}% explained variance)'.format(var_ratios[2]*100))

    plt.show()
