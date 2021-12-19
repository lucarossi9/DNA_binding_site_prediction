#### HELPER PREPROCESSING FOR BOTH REGRESSION AND CLASSIFICATION TASKS

# imports

from sklearn.ensemble import IsolationForest
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# FOR SPLITTING THE DATA

def split_importance(x, y, importance_boundary=0.7):
    """
  Split the samples into interesting ones and not interesting ones
  :param x: numpy.ndarray:the feature vector of the initial dataset
  :param y: numpy.ndarray: the value vector of the initial dataset
  :param importance_boundary: float: the lower bound for the underepresented class
  """
    return x[y >= importance_boundary], y[y >= importance_boundary], x[y < importance_boundary], y[
        y < importance_boundary]


## FOR OUTLIERS REMOTION

def split_outliers(threshold, scores):
    outliers_indices = np.where(scores < threshold)[0]
    return outliers_indices


def Anomaly_Detection_Isolation_Forests(x, change_split=False):
    """
    The function performs an outlier detection using random forest algorithm
    :param x: The set in which we would like to remove outliers
    :param change_split:
    :return: contamination :The amount of contamination of the data set, i.e. the proportion of outliers in the data set
             scores: The anomaly score of the input samples. The lower, the more abnormal.
             Negative scores represent outliers, positive scores represent inliers.
             outliers_indices: a vector of outliers indices
    """
    random_state = np.random.RandomState(42)
    contamination = 'auto'
    threshold = -0.025
    model = IsolationForest(n_estimators=120, max_samples='auto', contamination=contamination,
                            random_state=random_state)
    model.fit(x)
    scores = model.decision_function(x)
    if change_split == False:
        anomaly_score = model.predict(x)
        outliers_indices = np.where(anomaly_score == -1)[0]
    if change_split == True:
        outliers_indices = split_outliers(threshold, scores)
    return contamination, scores, outliers_indices


def check_Isolation_Forests(contamination, outliers_indices, relKa):
    """
    Simply a check on the proper working of the IF algorithm
    :param relKa: y labels
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set
    :param outliers_indices: the indices of the outliers
    :return: None
    """
    # we check that the difference between the contamination and the outliers percentage is close to zero
    tol: float = 1.0e-02
    if contamination != 'auto':
        outliers_percentage = 1 / len(relKa) * len(outliers_indices)
        assert np.abs(contamination - outliers_percentage) < tol


def check_boundary_decision(scores, p, verbose=1):
    """
  This function simply controls how many scores returned by the IF algorithm
  are likely to be misclassified
  :param scores: The anomaly score of the input samples. The lower, the more abnormal.
  :param p: a give threshold
  :param verbose: if verbose=1 we have the plot
  :return:
  """
    indecision_percentage = 1 / len(scores) * np.count_nonzero(np.abs(scores) <= p)
    if verbose == 1:
        plt.hist(scores)
        plt.xlabel('scores', fontsize=30)
        plt.ylabel('frequency', fontsize=30)
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        plt.show()
        print("The indecision percentage around", p, "is", indecision_percentage)
        print("The percentage of outliers detected is", 1 / len(scores) * len(np.where(scores < 0)[0]))


def drop_outliers(x, y, outliers):
    """
    This function simply remove the outliers from the x and y passed
    :param x: x_train
    :param y: y_train
    :param outliers: outliers indices
    :return:
    """
    x = np.delete(x, outliers, axis=0)
    y = np.delete(y, outliers, axis=0)
    return x, y
