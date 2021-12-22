# HELPER FOR THE METRICS

from sklearn.metrics import mean_squared_error as MSE
import scipy as sci
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import confusion_matrix


"""
Discussion: Before and after performing undersample and oversample of our dataset, we tried using a different loss to
train our model on in order to achieve better results in the minority class. As you can see from the final notebooks,
once the undersamping and oversampling were performed, there was no need to train our model with such metrics since we
have a dataset almost balanced (70% for the majority class and 30% of the minority class were the proportions which gave
us the best results). Therefore, this metrics were not used anymore to train the models but they were used only to
evaluate the goodness of fits of the models after training.
"""


# penalized loss for regression
def penalized_MSE(y_true, y_pred, decision_boundary=0.7):
    """
    The function computes an average loss between the point in the minority and the ones in the majority class
    :param y_true: true labels
    :param y_pred: predicted labels
    :param decision_boundary: The decision boundary between the two classes, a label bigger than the decision boundary
    is considered to be belonging to the minority class
    :return: loss = 1/2*MSE(minority class) + 1/2*MSE(majority class)
    """
    # we find the indices of the points in the minority class
    critical_indices = np.where(y_true >= decision_boundary)[0]

    # if no such indices were found or if all the points belong to such class we compute a simple MSE
    if critical_indices.shape[0] == 0 | critical_indices.shape[0] == y_true.shape[0]:
        return (1 / 2) * MSE(y_true, y_pred)
    else:
        # we divide the labels and the predictions based on the value of the
        critical_y = y_true[critical_indices]
        common_y = np.delete(y_true, critical_indices)
        critical_predictions = y_pred[critical_indices]
        common_predictions = np.delete(y_pred, critical_indices)
        return (1 / 2) * MSE(critical_y, critical_predictions) + (1 / 2) * MSE(common_y, common_predictions)


# penalized metric for classification
def weighted_balance_average(y_true, y_pred):
    """
    This function calculates the weighted balanced accuracy of a classification model
    @param y_true: the true labels
    @param y_pred: the predicted labels
    @return: the accuracy of the prediction
    """

    # we compute the number of samples in the minority class using the true labels
    ones_number = np.count_nonzero(y_true == 1.)

    # we compute the frequency of the majority and of the minority class in the true labels
    frequency_1 = ones_number / len(y_true)
    frequency_0 = 1 - frequency_1

    # we compute the sum of the inverse frequencies
    sum_inverse_frequencies = 1 / frequency_0 + 1 / frequency_1

    # we define the weights of our accuracy metric as the inverse of the frequency of a class times times a corrective
    # term so that the weights are summing up to 1
    weight_0 = 1 / (frequency_0 * sum_inverse_frequencies)
    weight_1 = 1 / (frequency_1 * sum_inverse_frequencies)

    # we calculate the standard confusion matrix, our metric is a balanced average of the trace elements of C
    C = confusion_matrix(y_true, y_pred)

    # the accuracy is weight_0 * accuracy_0 + weight_1 * accuracy_1
    return weight_0 * C[0, 0] / (C[0, 0] + C[0, 1]) + weight_1 * C[1, 1] / (C[1, 1] + C[1, 0])


# Useful function for evaluating the classification model
def return_accuracy(y_test, y_pred, verbose=1):
    """
    This function computes the standard accuracy of a prediction
    @param y_test: the true labels
    @param y_pred: the predicted labels
    @param verbose: verbose parameter, if verbose=1 we print more details
    @return: the accuracy of the prediction
    """
    # we compute the confusion matrix
    C = confusion_matrix(y_test, y_pred)

    # we compute the accuracy
    accuracy = np.trace(C) / len(y_test)

    if verbose == 1:
        # print all the elements of the matrix and the accuracies on the two classes
        print("The number of true negatives is:", C[0, 0])
        print("The number of false negatives is:", C[1, 0])
        print("The number of false positives is:", C[0, 1])
        print("The number of true positives is:", C[1, 1])
        print("The accuracy is:", accuracy)
        print("The accuracy on the 1's is ", 1 / np.sum(y_test == 1) * (np.sum(y_test == 1) - C[1, 0]))
        print("The accuracy on the 0's is ", 1 / np.sum(y_test == 0) * (np.sum(y_test == 0) - C[0, 1]))
    return accuracy


# Useful function for the plot of the curves of the losses to evaluate if we are overfitting or not
def score_metrics_regression(x_train, x_test, y_train, y_test, model):
    """
    Returns 2-2d arrays, each of the arrays has 1 row for each iteration, the first array contains the training losses
    for each iteration, the second array contains the testing losses for each iteration. This function will be useful to
    understand if the model chosen is overfitting or underfitting.
    @param x_train: numpy.ndarray containing the features of the training set
    @param x_test: numpy.ndarray containing the features of the test set
    @param y_train: numpy.ndarray containing the labels of the training set
    @param y_test: numpy.ndarray containing the labels of the test set
    @param model: the regression model we are evaluating
    """
    training_scores = []
    testing_scores = []
    # we fit the model
    model.fit(x_train, y_train, eval_metric="rmse", eval_set=[(x_train, y_train), (x_test, y_test)])

    # we take the results for the training and test set and we append to the lists
    training_scores.append(model.evals_result()['validation_0']['rmse'])
    testing_scores.append(model.evals_result()['validation_1']['rmse'])

    return np.array(training_scores).T, np.array(testing_scores).T
