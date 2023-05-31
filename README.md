# Pipelines in scikit-learn - Lab 

## Introduction 

In this lab, you will work with the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). The goal of this lab is not to teach you a new classifier or even show you how to improve the performance of your existing model, but rather to help you streamline your machine learning workflows using scikit-learn pipelines. Pipelines let you keep your preprocessing and model building steps together, thus simplifying your cognitive load. You will see for yourself why pipelines are great by building the same KNN model twice in different ways. 

## Objectives 

- Construct pipelines in scikit-learn 
- Use pipelines in combination with `GridSearchCV()`

## Import the data

Run the following cell to import all the necessary classes, functions, and packages you need for this lab. 


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')
```

Import the `'winequality-red.csv'` dataset and print the first five rows of the data.  


```python
# Import the data
df = None


# Print the first five rows

```

Use the `.describe()` method to print the summary stats of all columns in `df`. Pay close attention to the range (min and max values) of all columns. What do you notice? 


```python
# Print the summary stats of all columns

```

As you can see from the data, not all features are on the same scale. Since we will be using k-nearest neighbors, which uses the distance between features to classify points, we need to bring all these features to the same scale. This can be done using standardization. 



However, before standardizing the data, let's split it into training and test sets. 

> Note: You should always split the data before applying any scaling/preprocessing techniques in order to avoid data leakage. If you don't recall why this is necessary, you should refer to the **KNN with scikit-learn - Lab.** 

## Split the data 

- Assign the target (`'quality'` column) to `y` 
- Drop this column and assign all the predictors to `X` 
- Split `X` and `y` into 75/25 training and test sets. Set `random_state` to 42  


```python
# Split the predictor and target variables
y = None
X = None

# Split into training and test sets
X_train, X_test, y_train, y_test = None
```

## Standardize your data 

- Instantiate a `StandardScaler()` 
- Transform and fit the training data 
- Transform the test data 


```python
# Instantiate StandardScaler
scaler = None

# Transform the training and test sets
scaled_data_train = None
scaled_data_test = None

# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
scaled_df_train.head()
```

## Train a model 

- Instantiate a `KNeighborsClassifier()` 
- Fit the classifier to the scaled training data 


```python
# Instantiate KNeighborsClassifier
clf = None

# Fit the classifier

```

Use the classifier's `.score()` method to calculate the accuracy on the test set (use the scaled test data) 


```python
# Print the accuracy on test set

```

Nicely done. This pattern (preprocessing and fitting models) is very common. Although this process is fairly straightforward once you get the hang of it, **pipelines** make this process simpler, intuitive, and less error-prone. 

Instead of standardizing and fitting the model separately, you can do this in one step using `sklearn`'s `Pipeline()`. A pipeline takes in any number of preprocessing steps, each with `.fit()` and `transform()` methods (like `StandardScaler()` above), and a final step with a `.fit()` method (an estimator like `KNeighborsClassifier()`). The pipeline then sequentially applies the preprocessing steps and finally fits the model. Do this now.   

## Build a pipeline (I) 

Build a pipeline with two steps: 

- First step: `StandardScaler()` 
- Second step (estimator): `KNeighborsClassifier()` 



```python
# Build a pipeline with StandardScaler and KNeighborsClassifier
scaled_pipeline_1 = None
```

- Transform and fit the model using this pipeline to the training data (you should use `X_train` here) 
- Print the accuracy of the model on the test set (you should use `X_test` here) 


```python
# Fit the training data to pipeline


# Print the accuracy on test set

```

If you did everything right, this answer should match the one from above! 

Of course, you can also perform a grid search to determine which combination of hyperparameters can be used to build the best possible model. The way you define the pipeline still remains the same. What you need to do next is define the grid and then use `GridSearchCV()`. Let's do this now.

## Build a pipeline (II)

Again, build a pipeline with two steps: 

- First step: `StandardScaler()` named 'ss'.  
- Second step (estimator): `RandomForestClassifier()` named 'RF'. Set `random_state=123` when instantiating the random forest classifier 


```python
# Build a pipeline with StandardScaler and RandomForestClassifier
scaled_pipeline_2 = None
```

Use the defined `grid` to perform a grid search. We limited the hyperparameters and possible values to only a few values in order to limit the runtime. 


```python
# Define the grid
grid = [{'RF__max_depth': [4, 5, 6], 
         'RF__min_samples_split': [2, 5, 10], 
         'RF__min_samples_leaf': [1, 3, 5]}]
```

Define a grid search now. Use: 
- the pipeline you defined above (`scaled_pipeline_2`) as the estimator 
- the parameter `grid` 
- `'accuracy'` to evaluate the score 
- 5-fold cross-validation 


```python
# Define a grid search
gridsearch = None
```

After defining the grid values and the grid search criteria, all that is left to do is fit the model to training data and then score the test set. Do this below: 


```python
# Fit the training data


# Print the accuracy on test set

```

## Summary

See how easy it is to define pipelines? Pipelines keep your preprocessing steps and models together, thus making your life easier. You can apply multiple preprocessing steps before fitting a model in a pipeline. You can even include dimensionality reduction techniques such as PCA in your pipelines. In a later section, you will work on this too! 
