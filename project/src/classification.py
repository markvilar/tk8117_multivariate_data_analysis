import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import KFold
from mpl_toolkits import mplot3d

from typing import Tuple

from plotting import pca_inspection

def normalize_confusion_matrix(M: np.ndarray, Y):
    return M.astype('float') / np.sum(M, axis=1)

def analyze_results(Y_train: np.ndarray, Yhat_train: np.ndarray, probs_train: np.ndarray,
        Y_test: np.ndarray, Yhat_test: np.ndarray, probs_test, normalize: bool):
    confmat_train = confusion_matrix(Y_train, Yhat_train)
    confmat_test = confusion_matrix(Y_test, Yhat_test)
    vmin_train = 0
    vmax_train = np.max(confmat_train)
    vmin_test = 0
    vmax_test = np.max(confmat_test)
    if normalize:
        confmat_train = normalize_confusion_matrix(confmat_train, Y_train)
        confmat_test = normalize_confusion_matrix(confmat_test, Y_test)
        vmin_train, vmax_train = 0, 1
        vmin_test, vmax_test = 0, 1
    # Confusion matrix for training set
    plt.figure(1, figsize=(8,8))
    plt.imshow(confmat_train, interpolation='nearest', vmin=vmin_train, vmax=vmax_train)
    plt.title('Confusion matrix for training set')
    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    # Confusion matrix for test set
    plt.figure(2, figsize=(8,8))
    plt.imshow(confmat_test, interpolation='nearest', vmin=vmin_test, vmax=vmax_test)
    plt.title('Confusion matrix for test set')
    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    training_acc = np.mean(np.equal(Y_train, Yhat_train, dtype=int))
    training_loss = log_loss(Y_train, probs_train)
    test_acc = np.mean(np.equal(Y_test, Yhat_test, dtype=int))
    test_loss = log_loss(Y_test, probs_test)
    print('Training set:')
    print('Accuracy: {:.2f}'.format(training_acc*100))
    print('Loss: {:.4f}'.format(training_loss))
    print('Test set:')
    print('Accuracy: {:.2f}'.format(test_acc*100))
    print('Loss: {:.4f}'.format(test_loss))
    plt.show()

def linear_classification(X: np.ndarray, Y: np.ndarray, X_test: np.ndarray,
        Y_test: np.ndarray, n_folds: int, n_comps_max: int, threshold: float, show_plots: bool,
        fignum: int, figsize: Tuple[int, int], normalize: bool):
    # Create k-folds
    kf = KFold(n_splits=n_folds)
    # PCA - CV
    cum_var_ratios = np.zeros((n_folds, n_comps_max))
    for i, (train_inds, val_inds) in enumerate(kf.split(X)):
        X_train, X_val = X[train_inds,:], X[val_inds,:]
        model = decomposition.PCA(n_components=n_comps_max)
        scores = model.fit_transform(X_train)
        cum_var_ratios[i,:] = np.cumsum(model.explained_variance_ratio_)
    cum_var_ratios = np.pad(cum_var_ratios, ((0,0),(1,0)), 'constant')
    cum_var_means = np.mean(cum_var_ratios, axis=0)
    cum_var_stds = np.std(cum_var_ratios, axis=0)
    # Plot CV explained variance
    if show_plots:
        plt.figure(num=fignum, figsize=figsize)
        plt.errorbar(np.arange(0, n_comps_max+1), cum_var_means, yerr=cum_var_stds, ecolor='r')
        plt.title('Explained Variance ({:d}-fold CV)'.format(n_folds))
        plt.xlabel('PCs')
        plt.ylabel('Cumulative explained variance')
        plt.xlim(0, n_comps_max)
        plt.ylim(0, 1)
        plt.show()
    # Find number of components based on CV
    n_comps = np.where(cum_var_means>=threshold)[0][0]
    print('In linear analysis: ')
    print('# of PCs need to explain {:.0f}% variance in x: {}\n'.format(threshold*100, n_comps))
    # PCA model
    pca_model = decomposition.PCA(n_components=n_comps)
    train_scores = pca_model.fit_transform(X)
    if show_plots:
        pca_inspection(X, Y, n_comps)
    test_scores = pca_model.transform(X_test)
    # LDA model
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(train_scores, Y)
    Yhat_train = lda_model.predict(train_scores)
    probs_train = lda_model.predict_proba(train_scores)
    # Predict
    Yhat_test = lda_model.predict(test_scores)
    probs_test = lda_model.predict_proba(test_scores)
    # Result analysis
    analyze_results(Y, Yhat_train, probs_train, Y_test, Yhat_test, probs_test,
            normalize=normalize)

def svm_cross_validation(X: np.ndarray, Y: np.ndarray, n_folds: int, kernel: str, 
        gammas: np.ndarray, show_plots: bool, decision: str='ovr'):
    kf = KFold(n_splits=n_folds, shuffle=True)
    n_gammas = gammas.shape[0]
    val_accs = np.zeros((n_folds, n_gammas))
    val_losses = np.zeros((n_folds, n_gammas))
    for i, (train_inds, val_inds) in enumerate(kf.split(X)):
        X_train, Y_train = X[train_inds,:], Y[train_inds]
        X_val, Y_val = X[val_inds,:], Y[val_inds]
        for j, gamma in enumerate(gammas):
            model = SVC(kernel=kernel, gamma=gamma, probability=True, 
                    decision_function_shape=decision)
            model.fit(X_train, Y_train)
            Yhat_val = model.predict(X_val)
            probs_val = model.predict_proba(X_val)
            val_acc = np.mean(np.equal(Y_val, Yhat_val, dtype=int))
            val_loss = log_loss(labels=Y_val, y_true=Y_val, y_pred=probs_val)
            val_accs[i, j] = val_acc
            val_losses[i, j] = val_loss
    # CV accuracy and loss
    mean_accs = np.mean(val_accs, axis=0)
    std_accs = np.std(val_accs, axis=0)
    mean_losses = np.mean(val_losses, axis=0)
    std_losses = np.std(val_losses, axis=0)

    if show_plots:
        fig = plt.figure(1, (8,8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.errorbar(gammas, mean_accs, yerr=std_accs, ecolor='r')
        ax1.set_title('Accuracy ({:d}-fold CV)'.format(n_folds))
        ax1.set_ylabel('Accuracy [-]')
        ax1.set_xlabel('Gamma [-]')
        ax1.set_xlim(gammas[0], gammas[-1])
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.errorbar(gammas, mean_losses, yerr=std_losses, ecolor='r')
        ax2.set_title('Cross entropy loss ({:d}-fold CV)'.format(n_folds))
        ax2.set_ylabel('Cross entropy [-]')
        ax2.set_xlabel('Gamma [-]')
        ax2.set_xlim(gammas[0], gammas[-1])
        plt.show()

    best_gamma = gammas[np.argmin(mean_losses)]
    return best_gamma

def svm_classification(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
        Y_test: np.ndarray, kernel: str, gamma: float, decision: str='ovr'):
    model = SVC(kernel=kernel, gamma=gamma, probability=True, 
            decision_function_shape=decision)
    model.fit(X_train, Y_train)
    Yhat_train = model.predict(X_train)
    probs_train = model.predict_proba(X_train)
    Yhat_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)
    # Metrics
    acc_train = np.mean(np.equal(Y_train, Yhat_train, dtype=int))
    loss_train = log_loss(labels=Y_train, y_true=Y_train, y_pred=probs_train)
    acc_test = np.mean(np.equal(Y_test, Yhat_test, dtype=int))
    loss_test = log_loss(labels=Y_test, y_true=Y_test, y_pred=probs_test)
    print('Training set:')
    print('Accuracy: {:.2f}'.format(acc_train*100))
    print('Loss: {:.4f}'.format(loss_train))
    print('Test set:')
    print('Accuracy: {:.2f}'.format(acc_test*100))
    print('Loss: {:.4f}'.format(loss_test))
    confmat_train = confusion_matrix(Y_train, Yhat_train)
    confmat_test = confusion_matrix(Y_test, Yhat_test)
    vmin_train = 0
    vmax_train = np.max(confmat_train)
    vmin_test = 0
    vmax_test = np.max(confmat_test)
    # Confusion matrix for training set
    plt.figure(1, figsize=(8,8))
    plt.imshow(confmat_train, interpolation='nearest', vmin=vmin_train, vmax=vmax_train)
    plt.title('Confusion matrix for training set')
    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    # Confusion matrix for test set
    plt.figure(2, figsize=(8,8))
    plt.imshow(confmat_test, interpolation='nearest', vmin=vmin_test, vmax=vmax_test)
    plt.title('Confusion matrix for test set')
    plt.ylabel('True classes')
    plt.xlabel('Predicted classes')
    plt.show()
