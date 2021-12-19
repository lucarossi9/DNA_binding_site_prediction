# SAMPLING HELPER

# imports
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from scipy.stats import spearmanr


# For Undersampling

# quantization functions required for some score functions

def quantize(x, cuts=100):
    """
    The function transform the x in discrete values to later apply some of the oversampling algorithms
    :param x: feature vector
    :param cuts: number of bins in which we will cut the feature
    :return: the new x quantized
    """
    ranges = np.sort(np.unique(pd.cut(x, cuts)))
    for i in range(ranges.shape[0]):
        ranges_i = ranges[i]
        x[(x <= ranges_i.right) & (x > ranges_i.left)] = (ranges_i.left + ranges_i.right) / 2
    return x


def quantize_features(x, cuts=100):
    """
    This function apply the quantization along all the columns of the matrix x
    :param x: matrix of the features
    :param cuts: number of bins in which we will cut the feature
    :return: the new matrix quantized
    """
    return np.apply_along_axis(quantize, 0, x, cuts)


# correlation scorers
def Spearman(i, j):
    """
    The function computes spearman correlation score between the two vectors
    :param i: vector 1
    :param j: vector 2
    :return: correlation coefficient
    """
    ret, _ = spearmanr(i, j)
    return ret


def Correlation_Score_Min(x, y):
    """
    The function returns the minimum correlation score for each feature: for each feature it computes the minimum
    correlation score between this feature and all the others
    :param x: matrix of the features
    :param y: labels---> not used here but we had to pass it because it was required in SMOTE sampling
    :return: a vector containing for each feature the minimum correlation score between this feature and all the others
    """
    x = x.T
    score = [min([abs(Spearman(j, i)) for i in x]) for j in x]
    return np.array(score)


def Correlation_Score_Max(x, y):
    """
    The function returns the maximum correlation score for each feature (changing the sign): for each feature
    it computes the maximum correlation score between this feature and all the others and change his sign.
    :param x: matrix of the features
    :param y: labels---> not used here but we had to pass it because it was required in SMOTE sampling
    :return: a vector containing for each feature the maximum correlation score between this feature and all the others
    changing the sign
    """
    x = x.T
    score = [-max([abs(Spearman(j, i)) for i in x]) for j in x]
    return np.array(score)


def PSU_undersampling_regression(x, y, randomsize, xi, yi):
    """
    This function calculates the PSU undersampling of the samples in the x,y arrays
    with regards to the set xi,yi
    :param x: numpy.ndarray: The array of samples, which we want to undersample
    :param y: numpy.ndarray: The array with the labels corresponding to the samples in x
    :param randomsize: int: An int corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of these samples
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    C = np.mean(x, axis=0)
    dist = np.linalg.norm(x - C, 2, axis=1)
    indices = dist.argsort()
    x = x[indices]
    y = y[indices]
    split_x = np.array_split(x, randomsize)
    split_y = np.array_split(y, randomsize)
    indices = [Closest(split_x[i], xi) for i in range(randomsize)]

    x_resample = np.array([split_x[i][indices[i]] for i in range(randomsize)])
    y_resample = np.array([split_y[i][indices[i]] for i in range(randomsize)])
    return x_resample, y_resample


def Furthest(x, z):
    """
    Returns the index of the sample in x which is furthest from all the samples in z
    :param x: numpy.ndarray: The array of the samples for which we seek to find the furthest
    :param z: numpy.ndarray: The array of the samples which we use as a point of reference
    :return: int: returns a single index corresponding to the furthest x
    """
    max_dist_i = 0
    max_dist = -1
    for i in range(x.shape[0]):
        distance = min([np.linalg.norm(j - x[i]) for j in z])
        if max_dist < distance:
            max_dist = distance
            max_dist_i = i
    return max_dist_i


def Closest(x, z):
    """
    Returns the index of the sample in x which is furthest from all the samples in z
    :param x: numpy.ndarray: The array of the samples for which we seek to find the furthest
    :param z: numpy.ndarray: The array of the samples which we use as a point of reference
    :return: int: returns a single index corresponding to the furthest x
    """
    distances = []
    for i in range(x.shape[0]):
        distance = min([np.linalg.norm(j - x[i]) for j in z])
        distances.append(distance)
    return np.argmin(distances)


def PSU_undersampling(x, y, randomsize, xi, yi):
    """
    This function calculates the PSU undersampling of the samples in the x,y arrays
    with regards to the set xi,yi
    :param x: numpy.ndarray: The array of samples, which we want to undersample
    :param y: numpy.ndarray: The array with the labels corresponding to the samples in x
    :param randomsize: int: An int corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of these samples
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    C = np.mean(x, axis=0)
    dist = np.linalg.norm(x - C, 2, axis=1)
    indices = dist.argsort()
    x = x[indices]
    y = y[indices]
    split_x = np.array_split(x, randomsize)
    split_y = np.array_split(y, randomsize)
    dst_samples = [split_x[0][-1]]
    y_samples = [split_y[0][-1]]
    for i in range(1, randomsize):
        print("Starting ", i, " iteration")
        indice = Furthest(split_x[i], dst_samples)
        dst_samples.append(split_x[i][indice])
        y_samples.append(split_y[i][indice])
    x_resample = np.array(dst_samples)
    y_resample = np.array(y_samples)
    return x_resample, y_resample


def PSU_undersampling_reduced_dim(x, y, randomsize, xi, yi):
    """
    This function calculates the PSU undersampling by first doing a dimensionality reduction
    of the samples in the x,y arrays with regards to the set xi,yi
    :param x: numpy.ndarray: The array of samples, which we want to undersample
    :param y: numpy.ndarray: The array with the labels corresponding to the samples in x
    :param randomsize: int: An int corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of these samples
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    feature_scores = Fisher_Score(xi, x)
    indices = np.sort((-feature_scores).argsort()[:10])
    x_filtered = x[:, indices]
    x_i = xi[:, indices]
    C = np.mean(x_filtered, axis=0)
    dist = np.linalg.norm(x_filtered - C, 2, axis=1)
    indices = dist.argsort()
    x_filtered = x_filtered[indices]
    y = y[indices]
    split_x = np.array_split(x_filtered, randomsize)
    split_y = np.array_split(y, randomsize)
    indices = [Furthest(split_x[i], x_i) for i in range(randomsize)]
    split_x = np.array_split(x, randomsize)
    x_resample = np.array([split_x[i][indices[i]] for i in range(randomsize)])
    y_resample = np.array([split_y[i][indices[i]] for i in range(randomsize)])
    return x_resample, y_resample


def Fisher_Score(x_import, x_nimport):
    """
    Given two arrays of two classes this function calculates the Fischer_scores to
    measure the significance for all features
    :param x_import: numpy.ndarray: the array containing the samples of one class
    :param x_nimport: numpy.ndarray: the array containing the samples of the other class
    :return: numpy.ndarray: returns an array containg the Fisher_Score for all features
    """
    mean_import = np.mean(x_import, axis=0)
    mean_nimport = np.mean(x_nimport, axis=0)
    mean_dist = np.absolute(mean_import - mean_nimport)
    std_import = np.std(x_import, axis=0)
    std_nimport = np.std(x_nimport, axis=0)
    std_sum = std_import + std_nimport
    return np.divide(mean_dist, std_sum)


def calculate_distances(x, distance):
    """
    Calculates the distance between any two pairs of the set x using
    the Minkowski distance of degree distance.
    :param x: numpy.ndarray: the vector for which we will calculate the distance
                            between all of its elements
    :param distance: float: the norm which should be used for the Minkowski distance
    :return: numpy.ndarray: returns the Minkowski distance with the specified norm
                          between all pairs of elements in x
    """
    dist = np.array([[minkowski(a1, a2, distance) for a2 in x] for a1 in x])
    np.fill_diagonal(dist, float('inf'))
    return dist


def random_sampler(x, y, randomsize, x1, x2):
    """
    This function does random undersampling of vectors x,y and reduces them to
    size random size
    :param randomsize:
    :param x: numpy.ndarray: the feature vector to be subsampled
    :param y: numpy.ndarray: the label vector to be subsampled
    :return: <class 'tuple'>: A tuple containing the two undersampled vectors
    """
    p = np.random.permutation(len(y))
    new_x = x[p]
    new_y = y[p]
    return new_x[:randomsize], new_y[:randomsize]


def generate_samples(x, y, neighbors, N):
    """
    This function generate N samples which are convex combinations of
    the features of x and the labels of y
    :param N: number of samples to generate
    :param neighbors: neighbors of x
    :param x: numpy.ndarray:
    :param y: numpy.ndarray:
    :return: <class 'tuple'>:
    """
    new_samples_x = []
    new_samples_y = []
    for i in range(N):
        random_sample_i = random.randint(0, y.shape[0] - 1)
        x_i = x[random_sample_i]
        random_sample_j = random.randint(0, neighbors.shape[1] - 1)
        neigh_i = neighbors[random_sample_i, random_sample_j]
        x_j = x[neigh_i]
        lambda_ = random.uniform(0, 1)
        y_i = y[random_sample_i]
        y_j = y[neigh_i]
        new_x = x_i + lambda_ * (x_j - x_i)
        new_y = y_i + lambda_ * (y_j - y_i)
        new_samples_x.append(new_x)
        new_samples_y.append(new_y)
    return np.array(new_samples_x), np.array(new_samples_y)


def smote_sf(x, y, undersample=0.5, oversample=0.1, attribute_scorer=Fisher_Score,
             attribute_number=10, distance=float('inf'), kneighbors=3,
             undersampling=random_sampler, importance_class=0.7):
    """
    This function takes the complete input and produces a more balanced dataset based on the importance class
    :param x: numpy.ndarray: the feature vector of the initial dataset
    :param y: numpy.ndarray: the value vector of the initial dataset
    :param undersample: float: the percentage of the dominant class that we want to keep
    :param oversample: float: the percentage of the dataset that the small class will be at the end
    :param attribute_scorer: function: a function which will be used to score the relevance of a feature
    :param attribute_number: int: the number of attributes to keep according to their score
    :param distance: float: the norm which should be used for the Minkowski distance
    :param kneighbors: int: the number of samples which should be considered for each point
    :param undersampling: function: the function to use for the undersampling of the majority class
    :param importance_class: float: the lower bound for the underepresented class
    :return: returns 2 new feature vectors and 2 new label vectors containing
            the data for the importance class and the data for the non importance
            class and their labels.
    """
    x_import = x[y >= importance_class]
    y_import = y[y >= importance_class]
    x_nimport = x[y < importance_class]
    y_nimport = y[y < importance_class]

    feature_scores = attribute_scorer(x_import, x_nimport)
    # find the attribute_number highest coordinates of the feature_scores vector
    indices = np.sort((-feature_scores).argsort()[:attribute_number])
    x_import_filtered = x_import[:, indices]
    distances = calculate_distances(x_import_filtered, distance)
    # find the k lowest indices
    neighbors = np.array([np.sort(d.argsort()[:(kneighbors)]) for d in distances])
    # undersampling for the majority class
    nimport_len = int(undersample * y_nimport.shape[0])
    x_nimport, y_nimport = undersampling(x_nimport, y_nimport, nimport_len, x_import, y_import)
    # Calculate the number of samples to be generated
    N = int(oversample * (y_nimport.shape[0]) - y_import.shape[0])
    # Generate N new samples
    new_samples_x, new_samples_y = generate_samples(x_import, y_import, neighbors, N)
    # merge the new samples of the minority class with its old samples
    x_import = np.concatenate((x_import, new_samples_x))
    y_import = np.concatenate((y_import, new_samples_y))

    x_ret = np.concatenate((x_import, x_nimport))
    y_ret = np.concatenate((y_import, y_nimport))
    return x_ret, y_ret
