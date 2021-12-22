# Files
In this directory we have the following files:

## Best Run Classification.ipynb
This script generates our best model for the classification task, i.e. our best performance in classifying the DNA Sequences to be likely or not likely to be binding sites. <br>
We perform our pipeline of operations for the feature pre-processing: we first do outliers detection with Isolation Forests, then we proceed by balancing the dataset performing an ensemble of PSU Undersampling and SMOTE Oversampling on the training data. We then train a Random Forest Classifier as our baseline model and we perform the prediction on the test. In the end, we plot also the confusion matrix associated to our best result: it was the main goal of our job to classify well the minority class while maintaining a good behaviour on the majority one. We refer to the report to illustrate our big improvements in this area.

## Best Run Regression.ipynb
This script generates our best model for the regression task, i.e. our best performance in predicting the value of the RelKa $\in [0,1]$.<br>
We perform our pipeline of operations for the feature pre-processing: we first do outliers detection with Isolation Forests, then we proceed by balancing the dataset performing an ensemble of PSU Undersampling and SMOTE Oversampling on the training data. We then train an XGBoost Regressor as our baseline model and we perform the prediction on the test. 
In the end, we plot the distribution of the errors for different intervals of the RelKa value: it was one of the goals of our job to have a distribution of the error as uniform as possible. We refer to the report to illustrate our big improvements in this area.

## Helper Files
The Code directory also contains several helper files which hold functions that are essential for our project. Namely We have the following files:

#### preprocessing_helper.py
This file contains essential helper functions,required by the notebooks in the preprocessing phase:<br><br>
1.**split_importance(x, y, importance_boundary=0.7)**: This function is used to split the data into a representative, equally unbalanced train, test and validation set<br>
2.**split_RelKa(y,p)**: This function simply performs the splitting of the RelKa to labels 0 or 1 according to whether the original label is above or below the importance boundary p. <br>
3.**Anomaly_Detection_Isolation_Forests(x, change_split=False)**: This function performs an outlier detection using Isolation Forest Algorithm where the boundary decision may be user-defined by using the boolean change_split. <br>
4.**check_Isolation_Forests(contamination, outliers_indices, relKa)**: This functions simply checks the proper working of the Isolation Forest algorithm. <br>
5.**check_boundary_decision(scores, p, verbose=1)**: This functios controls how many scores returned by the Isolation Forest Algorithm are below some threshold and hence likely to be misclassified. <br>
6.**split_outliers(threshold, scores)**: This function is used to find the indices of the outliers in case the decision boundary of the outliers detection algorithm is changed. <br>
7.**drop_outliers(x, y, outliers)**: This function removes the outliers from the x and y passed. <br>
8.**one_hot_encode_sequence(seq)**: This function perform one hot encoding on a single DNA sequence, for each sequence it maps each character to a vector
    four dimensional vector of zeros and ones. <br>
9.**one_hot_encoding_df(df)**: This function performs one hot encoding on the column Kmer of the dataframe df. <br>
10.**sequential_encoding(df)**: This function performs the sequential encoding of an entire column of sequences of DNA in df by mapping the features to some predefined dictionary. <br>

#### sampling_helper.py
This file contains essential helper functions, required by the notebooks for undersampling and oversampling:<br><br>
1.**stratified_undersampling(x,y,intervals)**: This function performs stratified undersampling on the majority class by dropping samples in the i-th stratum with a probability equal to the frequency of that stratum datapoints. <br>
2.**PSU_undersampling_regression(x, y, randomsize, xi, yi)**:  This function performs a slight modification of the original PSU undersampling algorithm of the samples in the x,y arrays with respect to the set xi,yi. <br>
3.**PSU_undersampling(x, y, randomsize, xi, yi)** This function performs the standard PSU undersampling algorithm on the samples in the x,y arrays
  with respect to the set xi,yi. <br>
4.**PSU_stratified_undersampling(x, y, randomsize, intervals, xi, yi)**: This function performs stratified PSU undersampling of the samples in the x,y arrays
with regards to the set xi,yi. In particular the number of samples in the i-th stratum is rescaled by a factor equal to (1-f) where f is the frequency of that stratum datapoints. <br> 
5.**PSU_undersampling_reduced_dim(x, y, randomsize, xi, yi)** This function performs the modified PSU undersampling algorithm of the samples in the x,y arrays with regards to
  the set xi,yi by first doing a dimensionality reduction. <br>
6.**random_sampler(x, y, randomsize, x1, x2)** This function performs random undersampling of vectors x,y and reduces them to
  size random size. <br>
7.**Closest(x, z)** This function returns the index of the sample in x which is the closest one to all the samples in z. <br>
8.**Furthest(x, z)** This function returns the index of the sample in x which is furthest from all the samples in z. <br>
9.**generate_samples(x, y, neighbors, N)** This function generates N samples which are convex combinations of the features of x and the labels of y. <br>
10.**generate_samples_stratified(x, y, neighbors, N, intervals)**: This function generate N samples which are convex combinations of
the features of x and the labels of y, the samples are generated in a stratified way, we will try to oversample
more from the relKa intervals in which we have less parameters. <br>
11.**generate_samples_from_stratified(x, y, neighbors, N, indices, frequencies)**: This function generate N samples which are convex combinations of
    the features of x and the labels of y and is applied in each strata during the stratified version of SMOTE Oversampling. <br>
12.**quantize(x, cuts=100)** The function transform the x in discrete values to later apply some of the oversampling/undersampling algorithms<br>
13.**quantize_features(x, cuts=100)** This function applies the quantization along all the columns of the matrix x<br>
14.**Spearman(i, j)** The function computes spearman correlation score between the two vectors.<br>
15.**Correlation_score_Min(x, y)** This function computes for each feature the minimum correlation score between the latter and all the others. <br>
16.**Correlation_Score_Max(x, y)** This function computes for each feature the maximum correlation score between the latter and all the others, then changes the sign.<br>
17.**Fisher_Score(x_import, x_nimport)** This function calculates the Fisher score to measure the significance of the features.<br>
18.**calculate_distances(x, distance)** This function calculates the distance between any two pairs of the set x using the Minkowski distance of degree distance. <br>

19.**smote_sf_classification(x, y, undersample=0.1, oversample=0.3, attribute_scorer=Fisher_Score,
                            attribute_number=10, distance=float('inf'), kneighbors=3, importance_class=0.7)**: This function takes the complete input and produces a more balanced dataset based on the importance class for the classification task. In particular it performs PSU Undersampling on the majority class and SMOTE Oversampling on the minority one with the rates given by input. <br>
20.**smote_sf_regression(x, y, oversample=0.3, attribute_scorer=Fisher_Score,
                            attribute_number=10, distance=float('inf'), kneighbors=3, importance_class=0.7, width_strata=0.05)**: This function takes the complete input and produces a more balanced dataset based on the importance class for the regression task. In particular it performs stratified random undersampling on the majority class (so we pass also the width of the strata) and SMOTE oversampling on the minority one with the rate given in input. <br>


#### metrics_helper.py
This file contains functions essential for measuring the accuracy of our model, as we cannot rely on standard metrics without falling in the accuracy paradox. <br><br>
1.**penalized_MSE(y_true, y_pred, decision_boundary=0.7)**: This function computes an average loss between the point in the minority and the ones in the majority class. <br>
2.**weighted_balance_average(y_true, y_pred)**: This function calculates the weighted balanced accuracy of a classification model by taking a weighted average of the trace elements of the confusion matrix, the weights being defined as the rarity of each class to penalize more a false negative than a false positive. <br>
3.**return_accuracy(y_true, y_pred, verbose=1)** This function computes the accuracy of a prediction by taking the trace of the confusion matrix. It also prints if the verbose is active the accuracy on the majority and on the minority class respectively. <br>
4.**score_metrics_regression(x_train, x_test, y_train, y_test, iterations=3)** Returns 2-2d arrays, each of the arrays has 1 row for each iteration, the first array contains the training losses for each iteration, the second array contains the testing losses for each iteration. This function will be useful to
understand if the model chosen is overfitting or underfitting.<br>

