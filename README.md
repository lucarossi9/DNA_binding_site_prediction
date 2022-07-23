
Machine Learning project in collaboration with the [Laboratory for Computation and Visualization in Mathematics and Mechanics](https://www.epfl.ch/labs/lcvmm/).

A detailed report is available in the repository. 


# File Structure
```console
.
├── Code
│   ├── Best Run regression.ipynb
│   ├── Best Run classification.ipynb
│   ├── preprocessing_helper.py
│   ├── sampling_helper.py
│   ├── metrics_helper.py
│   ├── README.md
├── Data
│   └── README.md
├── README.md
└── Report.pdf

```
## Code

The code folder contains all the python files as well as all the jupyter notebooks. For more information read the corresponding **README.md** file in that directory.<br>
Here we will briefly present the optimal hyperparameters for the models that we use to get the best results.

### Optimal Parameters(Regression)
We used the **XGBoost Regressor** model with the following hyperparameters:
```python
parameters = {'subsample' : 0.8999999999999999,
              'n_estimators' : 500,
              'max_depth' : 20,
              'learning_rate' : 0.01,
              'colsample_bytree' : 0.7999999999999999,
              'colsample_bylevel' : 0.6}
```

### Optimal Parameters(Classification)
We used a **Random Forest Classifier** as our model with the following hyperparameters:

```python
parameters = {'n_estimators':400,
              'min_samples_split':5,
              'min_samples_leaf':1,
              'max_features':'auto',
              'max_depth':20,
              'bootstrap':False}
```

## Data

The data folder.<br>
It currently contains a link to the **data.zip** file containing the dataset of interest <br>
Open the zip inside that folder in order to execute the programs <br>
For more information read the corresponding **README.md** file in that directory.

### Report.pdf

The four pages report with an extensive explanation of our work, our thought process, our results and our conclusions.
