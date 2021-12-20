# SAMPLING HELPER

# imports
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import minkowski
from scipy.stats import spearmanr
import scipy.stats as st

# For Undersampling

# quantization functions required for some score functions


def quantize(x, cuts=100):
    """
    The function transform the x in discrete values to later apply some of the oversampling algorithms
    :param x: feature vector
    :param cuts: number of bins in which we will cut the feature
    :return: the new x quantized
    """
    # segment the array into discrete intervals
    ranges = np.sort(np.unique(pd.cut(x, cuts)))
    for i in range(ranges.shape[0]):
        ranges_i = ranges[i]
        # for all elements in array that belong to interval i set their values equal to the mean
        x[(x <= ranges_i.right) & (x > ranges_i.left)] = (ranges_i.left + ranges_i.right) / 2
    return x


def quantize_features(x, cuts=100):
    """
    This function apply the quantization along all the columns of the matrix x
    :param x: matrix of the features
    :param cuts: number of bins in which we will cut the feature
    :return: the new matrix quantized
    """
    #apply the quantization function
    return np.apply_along_axis(quantize, 0, x, cuts)


# CORRELATION SCORES
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


def stratified_undersampling(x, y, intervals):
    """
    This function performs stratified undersampling on the majority class
    by dropping samples in each strata with a probability equal to the frequency of the i-th stratum datapoints
    :param x: the datapoints to be undersampled
    :param y: the corresponding labels
    :param intervals: the intervals for stratification
    :return x and y undersampled
    """
    to_drop = []
    # we do a cycle over all intervals
    for i in range(len(intervals)-1):
        # get the indices for which the label is in the i-th strata
        indices = np.where(np.logical_and(intervals[i] < y, y < intervals[i+1]))[0]
        # compute the frequency of the i-th strata
        frequency = len(indices) / len(y)
        # we do a cycle over all elements in such a stratum
        for j in range(len(indices)):
            # we draw a uniform random variable
            U = st.uniform.rvs(size=1)
            if U <= frequency:
                # we drop every sample in the i-th strata with probability alpha * frequency
                to_drop.append(indices[j])
    # we drop from x and y the samples and the labels found before
    x_new = np.delete(x, to_drop, axis=0)
    y_new = np.delete(y, to_drop, axis=0)
    return x_new, y_new


# THIS FUNCTION IS NOT USED, OUTPERFORMED BY STRATIFIED UNDERSAMPLING
def PSU_undersampling_regression(x, y, randomsize, xi, yi):
    """
    This function performs the PSU undersampling of the samples in the x,y arrays
    with regards to the set xi,yi. This version is a slight modification of the original proposed algorithm
    which helps us improving in the prediction of the RelKa in the central intervals
    :param x: numpy.ndarray: The array of samples on which we perform the undersampling
    :param y: numpy.ndarray: The labels corresponding to the samples in x
    :param randomsize: int: An integer corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of the samples in xi
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    # we calculate the centroid of the majority class data
    C = np.mean(x, axis=0)
    # for each sample in the majority class we calculate the l2 distance from the centroid
    dist = np.linalg.norm(x-C, 2, axis=1)
    # we sort the distances
    indices = dist.argsort()
    # we sort the samples and the corresponding labels according to the distances calculated before
    x = x[indices]
    y = y[indices]
    # we split the samples and the corresponding labels in randomsize partitions
    split_x = np.array_split(x, randomsize)
    split_y = np.array_split(y, randomsize)
    # for each partition we calculate the closest point to the minority class xi
    indices = [Closest(split_x[i], xi) for i in range(randomsize)]
    # the collection of the randomsize closest points calculated above represents the undersampled data
    x_resample = np.array([split_x[i][indices[i]] for i in range(randomsize)])
    # the corresponding labels of the undersampled data are also returned
    y_resample = np.array([split_y[i][indices[i]] for i in range(randomsize)])
    return x_resample, y_resample


def PSU_stratified_undersampling(x, y, randomsize, intervals, xi, yi):
    """
    This function performs stratified PSU undersampling of the samples in the x,y arrays
    with regards to the set xi,yi. This version is a stratified version of the
    slightly modified original proposed algorithm
    :param x: numpy.ndarray: The array of samples on which we perform the undersampling
    :param y: numpy.ndarray: The labels corresponding to the samples in x
    :param randomsize: int: An integer corresponding to the number of samples to be left
    :param intervals: the intervals for stratification
    :param xi: the reference points
    :param yi: the corresponding labels to xi
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    # we calculate the number of strata
    k = len(intervals)
    # we initialize the lists for undersampling
    x_resample = np.zeros(x.shape[1]).T
    y_resample = []
    for i in range(len(intervals)-1):
        # we get the samples and the labels in the i-th strata
        indices = np.where(np.logical_and(intervals[i] < y, y < intervals[i+1]))[0]
        x_ = x[indices]
        y_ = y[indices]
        # we perform rescaled PSU undersampling on the i-th strata
        frequency = len(indices) / len(y)
        x_, y_ = PSU_undersampling_regression(x_, y_, np.int(randomsize * frequency), xi, yi)
        # we add the samples and the corresponding labels
        x_resample = np.concatenate([x_resample, x_], axis=0)
        y_resample = np.concatenate([y_resample, y_], axis=0)
    # we drop the first row used only for initialization
    x_resample = np.delete(x_resample, 0, axis=0)
    return x_resample, y_resample


def Furthest(x, z):
    """
    Returns the index of the sample in x which is furthest from all the samples in z
    :param x: numpy.ndarray: The array of the samples for which we seek to find the furthest
    :param z: numpy.ndarray: The array of the samples which we use as a point of reference
    :return: int: returns a single index corresponding to the furthest x
    """
    # we initialize the maximum distance to be -1 and the furthest sample to have index 0
    max_dist_i = 0
    max_dist = -1
    # we seek for the furthest point from z in x
    for i in range(x.shape[0]):
        # for each sample in x we calculate the distance from all points in z and we take the minimum
        distance = min([np.linalg.norm(j - x[i]) for j in z])
        # we update the distance if the current distance is larger than the old one
        if max_dist < distance:
            max_dist = distance
            # we save the current sample i as the candidate furthest point from z
            max_dist_i = i
    return max_dist_i


def Closest(x, z):
    """
    Returns the index of the sample in x which is the closest one to all the samples in z
    :param x: numpy.ndarray: The array of the samples for which we seek to find the furthest
    :param z: numpy.ndarray: The array of the samples which we use as a point of reference
    :return: int: returns a single index corresponding to the furthest x
    """
    distances = []
    # we seek for the closest point to z in x
    for i in range(x.shape[0]):
        # for each sample in x we calculate the distance from all points in z and we take the minimum
        distance = min([np.linalg.norm(j - x[i]) for j in z])
        # we add the calculated distance to the list
        distances.append(distance)
    # we return the closest point by taking the argmin over all calculated distances
    return np.argmin(distances)


# BEST PERFORMING FOR THE CLASSIFICATION
def PSU_undersampling(x, y, randomsize, xi, yi):
    """
    This function performs the standard PSU undersampling algorithm on the samples in the x,y arrays
    with regards to the set xi,yi
    :param x: numpy.ndarray: The array of samples to be undersampled
    :param y: numpy.ndarray: The labels corresponding to the samples in x
    :param randomsize: int: An integer corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of these samples
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    # we calculate the centroid of the majority class data
    C = np.mean(x, axis=0)
    # for each sample in the majority class we calculate the l2 distance from the centroid
    dist = np.linalg.norm(x - C, 2, axis=1)
    # we sort the distances
    indices = dist.argsort()
    # we sort the samples and the corresponding labels according to the distances calculated before
    x = x[indices]
    y = y[indices]
    # we split the samples and the corresponding labels in randomsize partitions
    split_x = np.array_split(x, randomsize)
    split_y = np.array_split(y, randomsize)
    # we initialize the undersampled data list with the last point in the first partition
    dst_samples = [split_x[0][-1]]
    y_samples = [split_y[0][-1]]
    # we now calculate the remaining randomsize-1 datapoints
    for i in range(1, randomsize):
        print("Starting ", i, " iteration")
        # we apply the following recursive scheme: we calculate the index of the furthest point in the i-th partition
        # from the set of the already selected i-1 points
        indice = Furthest(split_x[i], dst_samples)
        # we add the corresponding datapoint to be the i-th element of the list
        dst_samples.append(split_x[i][indice])
        # we add the corresponding label
        y_samples.append(split_y[i][indice])
    # we convert the two lists into arrays
    x_resample = np.array(dst_samples)
    y_resample = np.array(y_samples)
    return x_resample, y_resample


# NOT USED
def PSU_undersampling_reduced_dim(x, y, randomsize, xi, yi):
    """
    This function performs the modified PSU undersampling algorithm of the samples in the x,y arrays with regards to
     the set xi,yi by first doing a dimensionality reduction
    :param x: numpy.ndarray: The array of samples to be undersampled
    :param y: numpy.ndarray: The labels corresponding to the samples in x
    :param randomsize: int: An integer corresponding to the number of samples to be left
    :param xi: numpy.ndarray: The array with the samples to be used as a reference for distances
    :param yi: numpy.ndarray: The labels of these samples
    :return: numpy.ndarray,numpy.ndarray: Returns 2 numpy arrays corresponding to the undersampled data
    """
    # we calculate for each feature the Fischer score
    feature_scores = Fisher_Score(xi, x)
    # we sort the feature scores keeping only the 10 highest scored feature
    indices = np.sort((-feature_scores).argsort()[:10])
    # we sort the samples considering only the highest scored features
    x_filtered = x[:, indices]
    # we do this both on the majority and on the minority class
    x_i = xi[:, indices]
    # we calculate the centroid of the majority class data
    C = np.mean(x_filtered, axis=0)
    # for each sample in the majority class we calculate the l2 distance from the centroid
    dist = np.linalg.norm(x_filtered-C, 2, axis=1)
    # we sort the distances
    indices = dist.argsort()
    # we sort the samples and the corresponding labels according to the distances calculated before
    x_filtered = x_filtered[indices]
    y = y[indices]
    # we split the samples and the corresponding labels in randomsize partitions
    split_x = np.array_split(x_filtered, randomsize)
    split_y = np.array_split(y, randomsize)
    # for each partition we calculate the closest point to the minority class xi
    indices = [Furthest(split_x[i], x_i) for i in range(randomsize)]
    # the collection of the randomsize closest points calculated above represents the undersampled data
    x_resample = np.array([split_x[i][indices[i]] for i in range(randomsize)])
    # the corresponding labels of the undersampled data are also returned
    y_resample = np.array([split_y[i][indices[i]] for i in range(randomsize)])
    return x_resample, y_resample


def Fisher_Score(x_import, x_nimport):
    """
    Given two arrays this function calculates the Fisher score to measure the significance of the features
    :param x_import: numpy.ndarray: the array containing the samples of one class
    :param x_nimport: numpy.ndarray: the array containing the samples of the other class
    :return: numpy.ndarray: returns an array containg the Fisher_Score for all features
    """
    # we calculate the mean of the samples in the majority class
    mean_import = np.mean(x_import, axis=0)
    # we calculate the mean of the samples in the minority class
    mean_nimport = np.mean(x_nimport, axis=0)
    # we take the difference between the two quantities calculated before
    mean_dist = np.absolute(mean_import - mean_nimport)
    # we calculate the standard deviation of the samples in the majority class
    std_import = np.std(x_import, axis=0)
    # we calculate the standard deviation of the samples in the minority class
    std_nimport = np.std(x_nimport, axis=0)
    # we sum the two standard deviations calculated before
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
    # we calculate the distance between the datapoints in x
    dist = np.array([[minkowski(a1, a2, distance) for a2 in x] for a1 in x])
    # we fill the diagonal of the corresponding square matrix with infinity
    np.fill_diagonal(dist, float('inf'))
    return dist


def random_sampler(x, y, randomsize, x1, x2):
    """
    This function performs random undersampling of vectors x,y and reduces them to
    size random size
    :param randomsize: the size of the resulting undersampled data
    :param x: numpy.ndarray: the feature vector to be undersampled
    :param y: numpy.ndarray: the label vector to be undersampled
    :return: <class 'tuple'>: A tuple containing the two undersampled vectors
    """
    # we randomly permute x and y
    p = np.random.permutation(len(y))
    new_x = x[p]
    new_y = y[p]
    # we then return the first randomsize samples in x and the corresponding labels
    return new_x[:randomsize], new_y[:randomsize]


def generate_samples(x, y, neighbors, N):
    """
    This function generate N samples which are convex combinations of
    the features of x and the labels of y
    :param N: number of samples to generate
    :param neighbors: neighbors of x
    :param x: numpy.ndarray: the datapoints to be oversampled
    :param y: numpy.ndarray: the corresponding labels of the samples in x
    :return: <class 'tuple'>: a tuple containing the resampled data
    """
    new_samples_x = []
    new_samples_y = []
    for i in range(N):
        # we consider a random sample
        random_sample_i = random.randint(0, y.shape[0] - 1)
        x_i = x[random_sample_i]
        # we pick randomly one of its neighbours
        random_sample_j = random.randint(0, neighbors.shape[1] - 1)
        neigh_i = neighbors[random_sample_i, random_sample_j]
        x_j = x[neigh_i]
        # we draw a uniform random variable
        lambda_ = random.uniform(0, 1)
        # we get the label of the picked sample and of its neighbour
        y_i = y[random_sample_i]
        y_j = y[neigh_i]
        # we create the new sample and its label as a convex combination of the two selected ones
        new_x = x_i + lambda_ * (x_j - x_i)
        new_y = y_i + lambda_ * (y_j - y_i)
        # we append the new sample and its label to the two lists
        new_samples_x.append(new_x)
        new_samples_y.append(new_y)
    # we return two arrays containing the samples coming from the oversampling and their labels
    return np.array(new_samples_x), np.array(new_samples_y)


# FUNCTION WHICH PERFORMS UNDERSAMPLING AND OVERSAMPLING FOR THE CLASSIFICATION,
# THE UNDERSAMPLING TECHNIQUE IS THE PSU UNDERSAMPLING
def smote_sf_classification(x, y, undersample=0.1, oversample=0.3, attribute_scorer=Fisher_Score,
                            attribute_number=10, distance=float('inf'), kneighbors=3, importance_class=0.7):
    """
    This function takes the complete input and produces a more balanced dataset based on the importance class
    :param x: numpy.ndarray: the feature vector of the initial dataset
    :param y: numpy.ndarray: the value vector of the initial dataset
    :param undersample: float: the percentage of the dominant class that we want to keep
    :param oversample: float: the percentage of the dataset that the minority class will be at the end
    :param attribute_scorer: function: a function which will be used to score the relevance of a feature
    :param attribute_number: int: the number of attributes to keep according to their score
    :param distance: float: the norm which should be used for the Minkowski distance
    :param kneighbors: int: the number of nearest neighbours to be considered for each point
    :param importance_class: float: the lower bound for the under-represented class
    :return: returns 2 new feature vectors and 2 new label vectors containing
            the data for the importance class and the data for the non importance
            class and their labels.
    """
    # we split the training set into majority and minority class according to the value of the label
    x_import = x[y >= importance_class]
    y_import = y[y >= importance_class]
    x_nimport = x[y < importance_class]
    y_nimport = y[y < importance_class]

    # we calculate the Fisher score for all the features
    feature_scores = attribute_scorer(x_import, x_nimport)
    # we find the attribute_number highest coordinates of the feature_scores vector
    indices = np.sort((-feature_scores).argsort()[:attribute_number])
    # we filter the samples to be represented only by the attribute number highest scored features
    x_import_filtered = x_import[:, indices]
    # we calculate the distances between the datapoints considering only the important features as otherwise we
    # fall into the curse of dimensionality
    distances = calculate_distances(x_import_filtered, distance)
    # we find the kneighbors lowest indices of the distances of each sample, the corresponding datapoints are the
    # nearest neighbours to be drawn randomly for the oversampling procedure
    neighbors = np.array([np.sort(d.argsort()[:kneighbors]) for d in distances])
    # we first undersample the the majority class
    # we calculate the final size of the undersampled data
    nimport_len = int(undersample * y_nimport.shape[0])
    # we perform undersampling
    x_nimport, y_nimport = PSU_undersampling(x_nimport, y_nimport, nimport_len, x_import, y_import)
    # we calculate the number of samples to be generated by the oversampling algorithm
    N = int(oversample * (y_nimport.shape[0]) - y_import.shape[0])
    # we perform oversampling
    new_samples_x, new_samples_y = generate_samples(x_import, y_import, neighbors, N)
    # we merge the new samples with the old samples of the minority class and their labels
    x_import = np.concatenate((x_import, new_samples_x))
    y_import = np.concatenate((y_import, new_samples_y))
    # we stack the samples and the labels of the majority and minority class
    x_ret = np.concatenate((x_import, x_nimport))
    y_ret = np.concatenate((y_import, y_nimport))
    # x_ret represents the balanced dataset and y_ret the corresponding labels
    return x_ret, y_ret


# FUNCTION WHICH PERFORMS UNDERSAMPLING AND OVERSAMPLING FOR THE REGRESSION,
# THE STRATIFIED UNDERSAMPLING TECHNIQUE IS USED
def smote_sf_regression(x, y, undersample=0.5, oversample=0.1, attribute_scorer=Fisher_Score,
                        attribute_number=10, distance=float('inf'), kneighbors=3,
                        importance_class=0.7, width_strata=0.05):
    """
    This function performs stratified undersampling and SMOTE oversampling for the regression task
    @param x: the initial dataset
    @param y: the corresponding labels
    @param undersample: the undersampling rate
    @param oversample: the oversampling rate
    @param attribute_scorer: the scorer for the features
    @param attribute_number: the number of features to be considered for the distances between samples
    @param distance: the degree of the Minkowski distance to be used
    @param kneighbors: the number of nearest neighbours for the oversampling algorithm
    @param importance_class: the importance boundary between majority and minority class
    @param width_strata: the width of the strata for the stratified undersampling
    @return: x and y rebalanced
    """
    # we split the training set into majority and minority class according to the value of the label
    x_import = x[y >= importance_class]
    y_import = y[y >= importance_class]
    x_nimport = x[y < importance_class]
    y_nimport = y[y < importance_class]

    # we calculate the Fisher score for all the features
    feature_scores = attribute_scorer(x_import, x_nimport)
    # we find the attribute_number highest coordinates of the feature_scores vector
    indices = np.sort((-feature_scores).argsort()[:attribute_number])
    # we filter the samples to be represented only by the attribute number highest scored features
    x_import_filtered = x_import[:, indices]
    # we calculate the distances between the datapoints considering only the important features as otherwise we
    # fall into the curse of dimensionality
    distances = calculate_distances(x_import_filtered, distance)
    # we find the kneighbors lowest indices of the distances of each sample, the corresponding datapoints are the
    # nearest neighbours to be drawn randomly for the oversampling procedure
    neighbors = np.array([np.sort(d.argsort()[:kneighbors]) for d in distances])
    # we get the undersampled datasize
    nimport_len = int(undersample * y_nimport.shape[0])
    # we perform stratified undersampling
    intervals = width_strata * np.arange(np.int(importance_class / width_strata))
    x_nimport, y_nimport = stratified_undersampling(x_nimport, y_nimport, intervals)
    # we compute the number of new samples to be generated
    N = int(oversample * (y_nimport.shape[0]) - y_import.shape[0])
    new_samples_x, new_samples_y = generate_samples(x_import, y_import, neighbors, N)
    # we merge the new samples with the old samples of the minority class and their labels
    x_import = np.concatenate((x_import, new_samples_x))
    y_import = np.concatenate((y_import, new_samples_y))
    # we stack the samples and the labels of the majority and minority class
    x_ret = np.concatenate((x_import, x_nimport))
    y_ret = np.concatenate((y_import, y_nimport))
    # x_ret represents the balanced dataset and y_ret the corresponding labels
    return x_ret, y_ret
