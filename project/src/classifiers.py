import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import cross_decomposition 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from mpl_toolkits import mplot3d

def lda(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    # Fit model
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)
    # Predict
    Y_pred = model.predict(X_test)
    is_equal = np.equal(Y_pred, Y_test).astype(int)
    accuracy = np.mean(is_equal)
    # Confusion matrix
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    plt.figure()
    plt.imshow(conf_matrix)
    plt.show()

def svm():
    raise NotImplementedError
