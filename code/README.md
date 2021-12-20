# Files
In this directory we have the following files:

## Best Run Classification.py
This script generates our best model for the classification task, i.e. our best performance in classifying the DNA Sequences to be likely or not likely to be binding sites. <br>
We perform our pipeline of operations for the feature pre-processing: we first do outliers detection with Isolation Forests, then we proceed by balancing the dataset performing an ensemble of PSU Undersampling and SMOTE Oversampling on the training data. We then train a Random Forest Classifier as our baseline model and we perform the prediction on the test. In the end, we plot also the confusion matrix associated to our best result: it was the main goal of our job to classify well the minority class while maintaining a good behaviour on the majority one. We refer to the report to illustrate our big improvements in this area.

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
4.**score_metrics(x_train, x_test, y_train, y_test, iterations=3)** This function returns two 2d arrays, each of the arrays has 1 row for each iteration.<br>

