# HELPER FOR THE METRICS

from sklearn.metrics import mean_squared_error as MSE
import scipy as sci
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import confusion_matrix


# penalized losses

def penalized_MSE_helper(y_true, y_pred):
    """
    The function is a helper for the functions penalized_MSE and penalized_MSE_train, the function
    compute an average loss between the point in the minority and in the majority class
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: loss = 1/2*MSE(minority class) + 1/2*MSE(majority class)
    """
    critical_indices = np.where(y_true >= 0.7)[0]
    print(critical_indices)
    print(y_true, y_pred)
    if critical_indices.shape[0] == 0 | critical_indices.shape[0] == y_true.shape[0]:
        return (1 / 2) * MSE(y_true, y_pred)
    else:
        critical_y = y_true[critical_indices]
        common_y = np.delete(y_true, critical_indices)
        critical_predictions = y_pred[critical_indices]
        common_predictions = np.delete(y_pred, critical_indices)
        return (1 / 2) * MSE(critical_y, critical_predictions) + (1 / 2) * MSE(common_y, common_predictions)


def penalized_MSE(y_true, y_pred, fitted_lambda=-1.0008549422054305):
    """
    The function compute the penalized MSE for the test/val set, in case of the test we did not apply previously boxcox
    to y_test/y_val while we did for x_test, therefore we scale back the predictions with the inverse boxcox
    transformation and then we apply the helper function
    :param y_true: y_test/y_val
    :param y_pred: y_predicted from x_test or x_val
    :param fitted_lambda: the parameter lambda which comes from the boxcox transformation, necessary to scale back the
    y_pred
    :return: the penalized MSE for the test/val sets
    """
    return penalized_MSE_helper(y_true, sci.special.inv_boxcox(y_pred, fitted_lambda))


def penalized_MSE_train(y_true, y_pred, fitted_lambda=-1.0008549422054305):
    """
    The function compute the penalized MSE for the train set, in case of the train we applied previously boxcox to
    therefore we scale back the predictions and the true labels with the inverse boxcox transformation and
    then we apply the helper function
    :param y_true: y_train
    :param y_pred: y_predicted from x_train
    :param fitted_lambda: the parameter lambda which comes from the boxcox transformation, necessary to scale back the
    y_pred and the y_train
    :return: the penalized MSE for the train set
    """
    return penalized_MSE_helper(sci.special.inv_boxcox(y_true, fitted_lambda),
                                sci.special.inv_boxcox(y_pred, fitted_lambda))


# for error analysis

def score_metrics(x_train, x_test, y_train, y_test, iterations=3):
    """
    Returns 2-2d arrays, each of the arrays has 1 row for each iteration.
    @param x: numpy.ndarray
    @param y: numpy.ndarray
    @param model:dictionary
    @param iterations: int64
    """
    training_scores = []
    testing_scores = []
    for i in range(iterations):
        model = XGBRegressor(subsample=0.8999999999999999, n_estimators=500,
                             max_depth=20, learning_rate=0.01, colsample_bytree=0.7999999999999999,
                             colsample_bylevel=0.6, obj=penalized_MSE_train)
        y_train2 = sci.special.inv_boxcox(y_train, fitted_lambda)
        model.fit(x_train, y_train, eval_metric=penalized_MSE_helper, eval_set=[(x_train, y_train), (x_test, y_test)])
        training_scores.append(model.evals_result()['validation_0']['rmse'])
        testing_scores.append(model.evals_result()['validation_1']['rmse'])
        print(len(testing_scores))
        print(len(testing_scores[i]))
    return np.array(training_scores).T, np.array(testing_scores).T


def weighted_balance_average(y_true, y_pred):
    """
    This function calculates the weighted balanced accuracy of a classification model
    @param y_true: the true labels
    @param y_pred: the predicted labels
    @return: the accuracy of the prediction
    """
    # we compute the number of minority class samples in the true labels
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


def return_accuracy(y_test, y_pred, verbose=1):
    """
    This function computes the standard accuracy of a prediction
    @param y_test: the true labels
    @param y_pred: the predicted labels
    @param verbose: verbose parameter
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
