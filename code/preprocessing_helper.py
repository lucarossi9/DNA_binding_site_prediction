# HELPER PREPROCESSING FOR BOTH REGRESSION AND CLASSIFICATION TASKS

# imports

from sklearn.ensemble import IsolationForest
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

""" This notebook contains some functions used during the preprocessing of the dataset"""


# For splitting the data between majority and minority class
def split_importance(x, y, importance_boundary=0.7):
    """
  Split the samples into interesting ones and not interesting ones
  :param x: numpy.ndarray: the feature vector of the initial dataset
  :param y: numpy.ndarray: the label vector of the initial dataset
  :param importance_boundary: float: the lower bound for the under-represented class
  :return: in order the x and the y of the majority class, the x and the y of the minority class
  """
    # the function simply returns the x and the y of the majority class, the x and the y of the minority class
    return x[y >= importance_boundary], y[y >= importance_boundary], x[y < importance_boundary], y[y < importance_boundary]


# Used to transform the relKa in labels 0/1 for classification
def split_RelKa(y, p):
    """
    The function transforms the RelKa values in labels
    : param y: numpy.ndarray: the vector containing the relKa values
    : param p: float: the decision boundary, the class will be one if relKa>p, zero otherwise
    """
    # we simply return the list created with a Python comprehension
    return np.array([1 if value > p else 0 for value in y])


# for outliers detection
def Anomaly_Detection_Isolation_Forests(x, change_split=False, threshold=0):
    """
    The function performs an outlier detection using random forest algorithm
    :param x: The set in which we would like to remove outliers
    :param change_split: a parameter which changes the boundary decision of the Isolation Forest Algorithm, if True we
                         split between inliers and outliers according to the a user-choice threshold passed to the
                          function.
    :param threshold: the alternative threshold chosen by the user in case change_split=True
    :return: contamination : The amount of contamination of the data set, i.e. the proportion of expected outliers
                            in the data set.
             scores: The anomaly score of the input samples. The lower, the more abnormal. In the standard
                     Isolation Forests negative scores represent outliers, positive scores represent inliers.
             outliers_indices: the array containing the outliers indices.
    """
    random_state = np.random.RandomState(42)
    # the contamination parameter is set to 'auto' and is thus calculated
    # according to the criterion presented in the original paper about Isolation Forests
    # Otherwise it would be a user-defined float number corresponding to the expected percentage of outliers
    contamination = 'auto'

    # we create our Isolation Forests model setting the number of estimators and the other relevant parameters
    model = IsolationForest(n_estimators=120, max_samples='auto', contamination=contamination,
                            random_state=random_state)

    # we fit the model
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


# used to check the proper working of the Isolation Forest Algorithm
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


# used to check indecision percentage of the Isolation Forest Algorithm
def check_boundary_decision(scores, p, verbose=1):
    """
    This function controls the indecision percentage of the Isolation Forest Algorithm around its decision boundary
    :param scores: The anomaly score of the input samples. The lower, the more abnormal.
    :param p: a given threshold such that if the points have score below the threshold, they are considered outliers.
    :param verbose: if verbose=1 we plot the distribution of the scores.
    :return: None
    """
    # we calculate the indecision percentage as the amount of scores which are in absolute value below the threshold
    indecision_percentage = 1 / len(scores) * np.count_nonzero(np.abs(scores) <= p)
    if verbose == 1:

        # we plot the distribution of the scores
        plt.hist(scores)
        plt.xlabel('scores', fontsize=25)
        plt.ylabel('frequency', fontsize=25)
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20)
        plt.title('Distribution of the scores outliers')
        plt.show()

        # we print the indecision percentage and the percentage of outliers detected by the Isolation Forest Algorithm
        print("The indecision percentage around", p, "is", indecision_percentage)
        print("The percentage of outliers detected is", 1 / len(scores) * len(np.where(scores < 0)[0]))


# used to split the points in outliers or not
def split_outliers(threshold, scores):
    """
    The functions returns the indices of the outliers based on the scores of the Isolation Forest Algorithm: the scores
    below the threshold are classified as outliers, the scores above the threshold classified as non outliers.
    : param threshold: float containing the threshold.
    : param scores: numpy.ndarray: containing the score for each sample.
    """

    # we compute the outliers indices and return them
    outliers_indices = np.where(scores < threshold)[0]

    return outliers_indices


#used to drop outliers
def drop_outliers(x, y, outliers):
    """
    This function simply removes the outliers from the x and y passed
    :param x: the feature matrix of the dataset
    :param y: the array of labels of the dataset
    :param outliers: the outliers indices
    :return: x and y where the outliers have been removed
    """
    # we remove the features of the samples which were identified as outliers by the outlier detection algorithm
    x = np.delete(x, outliers, axis=0)

    # we remove the corresponding labels from the y vector
    y = np.delete(y, outliers, axis=0)
    return x, y

# FOR ENCODING OF THE DNA SEQUENCES (not using since the performances were slightly decreasing)

# for one hot encoding of a single feature
def one_hot_encode_sequence(seq):
    """
    The function perform one hot encoding on a single DNA sequence, for each sequence it maps each character to a vector
    four dimensional vector of zeros and ones.
    :param: seq: the DNA sequence
    :return: np.ndarray: a vector with the encoded sequence
    """
    mapping = dict(zip("ACGT", range(4)))
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

# for one hot encoding of an entire column of the dataframe
def one_hot_encoding_df(df):
    """
    The function performs one hot encoding on the column Kmer of the dataframe df
    :param: df a dataframe whose column Kmer contains the DNA sequence
    :return: the dataframe modified with the column encoded
    """
    # we apply the function one_hot_encode_sequence repeteadly using apply method
    sequences = df.Kmer.apply(lambda x: one_hot_encode(x).reshape(14 * 4))

    # we create a dataframe with the encodings
    df_to_merge = sequences.apply(pd.Series)

    # we merge it
    df = pd.merge(df, df_to_merge, right_index=True, left_index=True)

    # we drop the old column
    df.drop(columns=['Kmer'], inplace=True)
    return df

# for the sequential encoding of the entire column of the dataframe
def sequential_encoding(df):
    """
    The function performs the sequential encoding of an entire column of sequences of DNA in df, it maps the features
    as explained in the dictio below. For each sequence we will have a vector of dimension equal to the length of the
    sequence with values mapped as in the dictio below.
    :param: df: the dataframe whose column Kmer contains the DNA sequence
    :return: the dataframe modified with the column encoded
    """
    dictio = {
        'a': 0.25,
        'c': 0.50,
        'g': 0.75,
        't': 1.00
    }

    # we apply the encoding
    df_encoding = df['Kmer'].apply(lambda bps: pd.Series([dictio[bp] if bp in dictio else 0.0 for bp in bps.lower()]))

    # we merge the results
    df = pd.merge(df, df_encoding, left_index=True, right_index=True)

    return df