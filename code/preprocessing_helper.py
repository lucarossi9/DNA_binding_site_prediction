# HELPER PREPROCESSING FOR BOTH REGRESSION AND CLASSIFICATION TASKS

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
  :param importance_boundary: float: the lower bound for the under-represented class
  """
    return x[y >= importance_boundary], y[y >= importance_boundary], x[y < importance_boundary], y[
        y < importance_boundary]


# FOR OUTLIERS REMOTION

def split_outliers(threshold, scores):
    outliers_indices = np.where(scores < threshold)[0]
    return outliers_indices


def Anomaly_Detection_Isolation_Forests(x, change_split=False):
    """
    The function performs an outlier detection using random forest algorithm
    :param x: The set in which we would like to remove outliers
    :param change_split: a parameter which changes the boundary decision of the Isolation Forest Algorithm, if True we
                         split between inliers and outliers according to the a user-choice threshold
                         declared in split_outliers
    :return: contamination :The amount of contamination of the data set, i.e. the proportion of expected outliers
                            in the data set
             scores: The anomaly score of the input samples. The lower, the more abnormal. In the standard
                     Isolation Forests negative scores represent outliers, positive scores represent inliers.
             outliers_indices: the array containing the outliers indices
    """
    random_state = np.random.RandomState(42)
    # the contamination parameter is set to 'auto' and is thus calculated
    # according to the criterion presented in the original paper about Isolation Forests
    # Otherwise it would be a user-defined float number corresponding to the expected percentage of outliers
    contamination = 'auto'
    # the alternative threshold in case change_split = True
    threshold = -0.025
    # we create our Isolation Forests model setting the number of estimators and the other relevant parameters
    model = IsolationForest(n_estimators=120, max_samples='auto', contamination=contamination,
                            random_state=random_state)
    model.fit(x)
    # the decision function returns for each sample a score between [-1/2, 1/2]
    scores = model.decision_function(x)
    if not change_split:
        # if change_split = False, we apply the Isolation Forest with the usual threshold equal to 0
        anomaly_score = model.predict(x)
        # in particular, the function predict returns already -1 for outliers, 1 for inliers according to the criterion
        # specified above for the function scores
        outliers_indices = np.where(anomaly_score == -1)[0]
    if change_split:
        # we consider outliers the samples with a score below threshold
        outliers_indices = split_outliers(threshold, scores)
    return contamination, scores, outliers_indices


def check_Isolation_Forests(contamination, outliers_indices, RelKa):
    """
    This function simply checks the proper working of the Isolation Forests Algorithm: in particular if the
    contamination parameter is not 'auto', it checks that the outliers percentage found by the algorithm is equal to
    the contamination parameter up to a certain tolerance
    :param RelKa: y labels
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set
    :param outliers_indices: the indices of the outliers
    :return: None
    """
    # we check that the difference between the contamination and the outliers percentage is close to zero
    tol: float = 1.0e-02
    if contamination != 'auto':
        # we calculate the outliers percentage as the ratio between the outliers found and the number of samples
        outliers_percentage = 1 / len(RelKa) * len(outliers_indices)
        # we check that in absolute value the outliers percentage is at most tol distant from the contamination
        assert np.abs(contamination - outliers_percentage) < tol


def check_boundary_decision(scores, p, verbose=1):
    """
    This function controls the indecision percentage of the Isolation Forest Algorithm around its decision boundary
    :param scores: The anomaly score of the input samples. The lower, the more abnormal.
    :param p: a given threshold
    :param verbose: if verbose=1 we plot the distribution of the scores
    :return: None
    """
    # we calculate the indecision percentage as the amount of scores which are in absolute value below the threshold
    indecision_percentage = 1 / len(scores) * np.count_nonzero(np.abs(scores) <= p)
    if verbose == 1:
        # we plot the distribution of the scores
        plt.hist(scores)
        plt.xlabel('scores', fontsize=30)
        plt.ylabel('frequency', fontsize=30)
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        plt.show()
        # we print the indecision percentage and the percentage of outliers detected by the Isolation Forest Algorithm
        print("The indecision percentage around", p, "is", indecision_percentage)
        print("The percentage of outliers detected is", 1 / len(scores) * len(np.where(scores < 0)[0]))


def drop_outliers(x, y, outliers):
    """
    This function simply removes the outliers from the x and y passed
    :param x: x_train
    :param y: y_train
    :param outliers: the outliers indices
    :return: x and y where the outliers have been removed
    """
    # we remove the samples which were identified as outliers by the outlier detection algorithm
    x = np.delete(x, outliers, axis=0)
    # we remove the corresponding labels from the y vector
    y = np.delete(y, outliers, axis=0)
    return x, y


def split_RelKa(y, p):
    return np.array([1 if value > p else 0 for value in y])