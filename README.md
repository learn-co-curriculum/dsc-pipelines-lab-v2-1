
# Pipelines in scikit-learn - Lab 

## Introduction 

In this lab you will work with the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality). The goal of this lab is not to teach you a new classifier or even show you how to improve the performace of your existing model, but rather to help you streamline your machine learning workflows using scikit-learn pipelines. Pipelines let you keep your preprocessing and model building steps together, thus simplifying your cognitive load. You will see for yourself why pipelines are great by building the same KNN model twice, but in different ways. 

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


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
# Import the data
df = pd.read_csv('winequality-red.csv')

# Print the first five rows
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Use the `.describe()` method to print the summary stats of all columns in `df`. Pay close attention to the range (min and max values) of all columns. What do you notice? 


```python
# Print the summary stats of all columns

```


```python
# __SOLUTION__ 
# Print the summary stats of all columns
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



As you can see from the data, not all features are on the same scale. Since we will be using k-nearest neighbors, which uses distance between features to classify points, we need to bring all these features to the same scale. This can be done using standardization. 



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


```python
# __SOLUTION__ 
# Split the predictor and target variables
y = df['quality']
X = df.drop('quality', axis=1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
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


```python
# __SOLUTION__ 
# Instantiate StandardScaler
scaler = StandardScaler()

# Transform the training and test sets
scaled_data_train = scaler.fit_transform(X_train)
scaled_data_test = scaler.transform(X_test)

# Convert into a DataFrame
scaled_df_train = pd.DataFrame(scaled_data_train, columns=X_train.columns)
scaled_df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.974181</td>
      <td>-0.232603</td>
      <td>1.114588</td>
      <td>-0.246318</td>
      <td>-0.110746</td>
      <td>-1.060007</td>
      <td>-0.962240</td>
      <td>1.756955</td>
      <td>-0.786419</td>
      <td>-1.313194</td>
      <td>-1.152577</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.281894</td>
      <td>0.378026</td>
      <td>0.090887</td>
      <td>-0.246318</td>
      <td>0.193294</td>
      <td>-1.060007</td>
      <td>-0.962240</td>
      <td>1.105315</td>
      <td>0.316104</td>
      <td>-0.970646</td>
      <td>-1.247037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.710137</td>
      <td>0.322515</td>
      <td>-1.393481</td>
      <td>-0.317176</td>
      <td>0.051409</td>
      <td>-0.669757</td>
      <td>-0.992531</td>
      <td>-1.023376</td>
      <td>0.705229</td>
      <td>-0.628099</td>
      <td>1.019988</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.009880</td>
      <td>0.044956</td>
      <td>-0.165039</td>
      <td>0.603976</td>
      <td>-0.252631</td>
      <td>0.013182</td>
      <td>1.976031</td>
      <td>0.453675</td>
      <td>-0.267585</td>
      <td>-0.285551</td>
      <td>-0.963659</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.573668</td>
      <td>1.349482</td>
      <td>-0.011484</td>
      <td>0.178829</td>
      <td>-0.212093</td>
      <td>0.793683</td>
      <td>0.279710</td>
      <td>0.888102</td>
      <td>-0.008168</td>
      <td>0.056996</td>
      <td>0.169854</td>
    </tr>
  </tbody>
</table>
</div>



## Train a model 

- Instantiate a `KNeighborsClassifier()` 
- Fit the classifier to the scaled training data 


```python
# Instantiate KNeighborsClassifier
clf = None

# Fit the classifier

```


```python
# __SOLUTION__ 
# Instantiate KNeighborsClassifier
clf = KNeighborsClassifier()

# Fit the classifier
clf.fit(scaled_data_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                         weights='uniform')



Use the classifier's `.score()` method to calculate the accuracy on the test set (use the scaled test data) 


```python
# Print the accuracy on test set

```


```python
# __SOLUTION__ 
# Print the accuracy on test set
clf.score(scaled_data_test, y_test)
```




    0.5775



Nicely done. This pattern (preprocessing and fitting models) is very common. Although this process is fairly straightforward once you get the hang of it, **pipelines** make this process simpler, intuitive, and less error-prone. 

Instead of standardizing and fitting the model separately, you can do this in one step using `sklearn`'s `Pipeline()`. A pipeline takes in any number of preprocessing steps each with `.fit()` and `transform()` methods (like `StandardScaler()` above), and a final step with a `.fit()` method (an estimator like `KNeighborsClassifier()`). The pipeline then sequentially applies preprocessing steps and finally fits the model. Do this now.   

## Build a pipeline (I) 

Build a pipeline with two steps: 

- First step: `StandardScaler()` 
- Second step (estimator): `KNeighborsClassifier()` 



```python
# Build a pipeline with StandardScaler and KNeighborsClassifier
scaled_pipeline_1 = None
```


```python
# __SOLUTION__ 
# Build a pipeline with StandardScaler and KNeighborsClassifier
scaled_pipeline_1 = Pipeline([('ss', StandardScaler()), 
                              ('knn', KNeighborsClassifier())])
```

- Transform and fit the model using this pipeline to the training data (you should use `X_train` here) 
- Print the accuracy on test set (you should use `X_test` here) 


```python
# Fit the training data to pipeline


# Print the accuracy on test set

```


```python
# __SOLUTION__ 
# Fit the training data to pipeline
scaled_pipeline_1.fit(X_train, y_train)

# Print the accuracy on test set
scaled_pipeline_1.score(X_test, y_test)
```




    0.5775



If you did everything right, this answer should match the one from above! 

Of course, you can also perform a grid search to determine which combination of hyperparameters can be used to build the best possible model. The way you define the pipeline still remains the same. What you need to do next is define the grid and then use `GridSearchCV()`. Let's do this now.

## Build a pipeline (II)

Again, build a pipeline with two steps: 

- First step: `StandardScaler()` 
- Second step (estimator): `RandomForestClassifier()`. Set `random_state=123` when instantiating the random forest classifier 


```python
# Build a pipeline with StandardScaler and RandomForestClassifier
scaled_pipeline_2 = None
```


```python
# __SOLUTION__ 
# Build a pipeline with StandardScaler and RandomForestClassifier
scaled_pipeline_2 = Pipeline([('ss', StandardScaler()), 
                              ('RF', RandomForestClassifier(random_state=123))])
```

Use the defined `grid` to perform a grid search. We limited the hyperparameters and possible values to only a few values in order to limit the runtime. 


```python
# Define the grid
grid = [{'RF__max_depth': [4, 5, 6], 
         'RF__min_samples_split': [2, 5, 10], 
         'RF__min_samples_leaf': [1, 3, 5]}]
```


```python
# __SOLUTION__ 
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


```python
# __SOLUTION__ 
# Define a grid search
gridsearch = GridSearchCV(estimator=scaled_pipeline_2, 
                          param_grid=grid, 
                          scoring='accuracy', 
                          cv=5)
```

After defining the grid values and the grid search criteria, all that is left to do is fit it to training data and then score the test set. Do it below: 


```python
# Fit the training data


# Print the accuracy on test set

```


```python
# __SOLUTION__ 
# Fit the training data
gridsearch.fit(X_train, y_train)

# Print the accuracy on test set
gridsearch.score(X_test, y_test)
```




    0.6125



## Summary

See how easy it is to define pipelines? Pipelines keep your preprocessing steps and models together, thus making your life easier. You can apply multiple preprocessing steps before fitting a model in a pipeline. You can even include dimensionality reduction techniques such as PCA in your pipelines. In a later section, you will work on this too! 
