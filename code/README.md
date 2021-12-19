# Files
In this directory we have the following files:

## Best Run Regression.py
This script generates our best model for the regression task, i.e. our best performance in predicting the value of the RelKa $\in [0,1]$.<br>
We perform our pipeline of operations for the feature pre-processing: we first do outliers detection with Isolation Forests, then we proceed by balancing the dataset performing an ensemble of PSU Undersampling and SMOTE Oversampling on the training data. We then train an XGBoost Regressor as our baseline model and we perform the prediction on the test. 
In the end, we plot the distribution of the errors for different intervals of the RelKa value: it was one of the goals of our job to have a distribution of the error as uniform as possible. We refer to the report to illustrate our big improvements in this area.

## Helper Files
The Code directory also contains several helper files which hold functions that are essential for our project. Namely We have the following files:

#### preprocessing_helper.py
This file contains essential helper functions,required by the notebooks in the preprocessing phase:<br><br>
1.**split_importance(x, y, importance_boundary=0.7)**: This function is used to split the data into a representative, equally unbalanced train, test and validation set<br>
2.**split_outliers(threshold, scores)**: This function is used to find the indices of the outliers in case the decision boundary of the outliers detection algorithm is changed. <br>
3.**Anomaly_Detection_Isolation_Forests(x, change_split=False)**: This function performs an outlier detection using Isolation Forest Algorithm where the boundary decision may be user-defined by using the boolean change_split. <br>
4.**check_Isolation_Forests(contamination, outliers_indices, relKa)**: This functions simply checks the proper working of the Isolation Forest algorithm. <br>
5.**check_boundary_decision(scores, p, verbose=1)**: This functios controls how many scores returned by the Isolation Forest Algorithm are below some threshold and hence likely to be misclassified. <br>
6.**drop_outliers(x, y, outliers)**: This function removes the outliers from the x and y passed. <br>


#### sampling_helper.py
This file contains essential helper functions, required by the notebooks for undersampling and oversampling:<br><br>
1.**quantize(x, cuts=100)** The function transform the x in discrete values to later apply some of the oversampling/undersampling algorithms<br>
2.**quantize_features(x, cuts=100)** This function applies the quantization along all the columns of the matrix x<br>
3.**Spearman(i, j)** The function computes spearman correlation score between the two vectors.<br>
4.**Correlation_score_Min(x, y)** This function computes for each feature the minimum correlation score between the latter and all the others. <br>
5.**Correlation_Score_Max(x, y)** This function computes for each feature the maximum correlation score between the latter and all the others, then changes the sign.<br>
6.**PSU_undersampling_regression(x,y, randomsize, xi, yi)** This function performs a slight modification of the original PSU undersampling algorithm of the samples in the x,y arrays with respect to the set xi,yi. <br>
7.**Furthest(x, z)** This function returns the index of the sample in x which is furthest from all the samples in z. <br>
8.**Closest(x, z)** This function returns the index of the sample in x which is the closest one to all the samples in z. <br>
9.**PSU_undersampling(x, y, randomsize, xi, yi)** This function performs the standard PSU undersampling algorithm on the samples in the x,y arrays
    with respect to the set xi,yi. <br>
10.**PSU_undersampling_reduced_dim(x, y, randomsize, xi, yi)** This function performs the modified PSU undersampling algorithm of the samples in the x,y arrays with regards to
     the set xi,yi by first doing a dimensionality reduction. <br>
11.**Fisher_Score(x_import, x_nimport)** This function calculates the Fisher score to measure the significance of the features.<br>
12.**calculate_distances(x, distance)** This function calculates the distance between any two pairs of the set x using the Minkowski distance of degree distance. <br>
13.**random_sampler(x, y, randomsize, x1, x2)** This function performs random undersampling of vectors x,y and reduces them to
    size random size. <br>
14.**generate_samples(x, y, neighbors, N)** This function generates N samples which are convex combinations of the features of x and the labels of y. <br>
15.**smote_sf(x, y, undersample=0.5, oversample=0.1, attribute_scorer=Fisher_Score,
             attribute_number=10, distance=float('inf'), kneighbors=3,
             undersampling=random_sampler, importance_class=0.7)** This function takes the complete input and produces a more balanced dataset based on the importance class. <br>


#### metrics_helper.py
This file contains functions essential for measuring the accuracy of our model, as we cannot rely on standard metrics without falling in the accuracy paradox. <br><br>
1.**penalized_MSE_helper(y_true, y_pred)** The function computes an average loss between the point in the minority and in the majority class, it is a helper for the functions penalized_MSE and penalized_MSE_train. <br>
2.**penalized_MSE(y_true, y_pred, fitted_lambda=-1.0008549422054305)** The function compute the penalized MSE for the test/val set, in case of the test we did not apply previously a Box-Cox Transformation to y_test/y_val while we did for x_test, therefore we scale back the predictions with the inverse boxcox transformation and then we apply the helper function. <br>
3.**penalized_MSE_train(y_true, y_pred, fitted_lambda=-1.0008549422054305)** This function computes the penalized MSE for the train set, in case of the train we applied      previously a Box-Cox Transformation therefore we scale back the predictions and the true labels with the inverse boxcox transformation and then we apply the helper function. <br>
4.**score_metrics(x_train, x_test, y_train, y_test, iterations=3)** This function returns two 2d arrays, each of the arrays has 1 row for each iteration.

#### proj1_ridge_regress.py
This file contains functions essential for the hyperparameter fitting of our models for ridge regression:<br><br>
1.**cross_validation_ridge** This function conducts cross validation for given values of hyperparameters for ridge regression.<br>
2.**finetune_ridge** This function takes a hyperparameter vector for the degree and a vector for the lambda and finds the best hyperparameters out of the given ones<br>
3.**optimal_weights_ridge** This function ,given all the hyperparameters and the training samples, computes the optimal weights for this method.<br>
4.**predict_ridge** This function, given some weights and an augmentation degree, makes a prediction for the linear model.

#### proj1_logistic.py
This file contains functions essential for the hyperparameter fitting of our models for logistic regression:<br><br>
1.**cross_validation_logistic** This function conducts cross validation for given values of hyperparameters for regularized logistic regression.<br>
2.**finetune_logistic** This function takes a hyperparameter vector for the degree and a vector for the lambda and finds the best hyperparameters out of the given ones.<br>
3.**optimal_weights_logistic** This function, given all the hyperparameters and the training samples, computes the optimal weights for this method.<br>
4.**predict_logistic**This function, given some weights and an augmentation degree, makes a prediction for the logistic model.<br>
5.**reg_logistic_regression_plot** Given a test set and a train set the function fits the regularized logistic regression parameters on the train set and it calculates the loss over each iteration for both the test set and the train set. Then it returns the two loss vectors.<br>
6.**calculate_batch_gradient** Calculate the logistic regression gradient for a batch.(**Not used**)<br>
7.**learning_by_penalized_batch_gradient** The implementation for regularized logistic regression using the gradient descent method with a batch for each iteration instead of the whole dataset.(**Not used**)<br>
8.**cross_validation_logistic_batch**This function conducts cross validation for given values of hyperparameters for regularized logistic regression, using the batch gradient descent method.(**Not used**)<br>
9.**finetune_batch_logistic**This function takes a hyperparameter vector for the degree and a vector for the lambda and finds the best hyperparameters out of the given ones. It uses a batch gradient descent method.(**Not used**)<br>
10.**calculate Hessian** Calculates the Hessian of the logistic function loss(**Not used**)<br>
11.**logistic_regression_compute** Returns the loss, the gradient and the Hessian(**Not used**)<br>
12.**learning_by_newton_method** One step of newton method returns new weights and current loss(**Not used**)<br>
13.**penalized_logistic_regression** Calculates the loss, the gradient and the Hessian for the regularized model.(**Not used**)

## Notebook Executions
The code directory also contains seperate notebooks, one for each function we were required to implement.By executing the corresponding notebook you can see the hyperparameter fitting for that process and you will also create the corresponding output file in the **Data** folder.Namely we have the following files:<br>
1.**run_grad.ipynb** Executes a notebook that fits the parameters with the Gradient Descent Method for the linear model and makes the relative predictions.<br>
2.**run_stochastic_grad.ipynb** Executes a notebook that fits the parameters with the Stochastic Gradient Descent Method for the linear model and makes the relative predictions.<br>
3.**run_least_squares.ipynb** Executes a notebook that fits the parameters with the Least Squares Method for the linear model and makes the relative predictions.<br>
4.**run_ridge.ipynb** Executes a notebook that fits the parameters with the Ridge Regression for the linear model and makes the relative predictions.<br>
5.**run_logistic.ipynb** Executes a notebook that fits the parameters for Logistic Regression with the Gradient Descent for the logistic model and makes the relative predictions.<br>
6.**run_regularized_logistic.ipynb** Executes a notebook that fits the parameters for Regularized Logistic Regression with the Gradient Descent Method for the logistic model and makes the relative predictions.<br>
