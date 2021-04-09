# Decision Trees

### Concepts 
You're given a dataset of **30** elements, 15 of which belong to a positive class (denoted by `+` ) and 15 of which do not (denoted by `-`). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows that if we split by attribute A there are 13 items of the positive class and 2 of the negative class in one branch and 2 of the positive and 13 of the negative in the other branch. The right hand side shows that if we split the data by attribute B there are 8 items of the positive class and 7 of the negative class in one branch and 7 of the positive and 8 of the negative in the other branch.

<img src="images/decision_stump.png">

### 1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node? 

It may be helpful to discuss splitting criteria.


```python
"""
Attribute A generates the best split for the data. 
The best attribute to split a tree at each node is selected by considering 
the attribute that creates the purest child nodes. Gini impurity and information 
gain are two criteria that can be used to measure the quality of a split.
"""
```

### Decision Trees for Regression 

In this section, you will use decision trees to fit a regression model to the Combined Cycle Power Plant dataset. 

This dataset is from the UCI ML Dataset Repository, and has been included in the `data` folder of this repository as an Excel `.xlsx` file, `Folds5x2_pp.xlsx`. 

The features of this dataset consist of hourly average ambient variables taken from various sensors located around a power plant that record the ambient variables every second.  
- Temperature (AT) 
- Ambient Pressure (AP) 
- Relative Humidity (RH)
- Exhaust Vacuum (V) 

The target to predict is the net hourly electrical energy output (PE). 

The features and target variables are not normalized.

In the cells below, we import `pandas` and `numpy` for you, and we load the data into a pandas DataFrame. We also include code to inspect the first five rows and get the shape of the DataFrame.


```python
import pandas as pd 
import numpy as np 

# Load the data
filename = 'data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```python
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
      <th>AT</th>
      <th>V</th>
      <th>AP</th>
      <th>RH</th>
      <th>PE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.96</td>
      <td>41.76</td>
      <td>1024.07</td>
      <td>73.17</td>
      <td>463.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25.18</td>
      <td>62.96</td>
      <td>1020.04</td>
      <td>59.08</td>
      <td>444.37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.11</td>
      <td>39.40</td>
      <td>1012.16</td>
      <td>92.14</td>
      <td>488.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.86</td>
      <td>57.32</td>
      <td>1010.24</td>
      <td>76.64</td>
      <td>446.48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.82</td>
      <td>37.50</td>
      <td>1009.23</td>
      <td>96.62</td>
      <td>473.90</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (9568, 5)



Before fitting any models, you need to create training and testing splits for the data.

Below, we split the data into features and target ('PE') for you. 


```python
X = df[df.columns.difference(['PE'])]
y = df['PE']
```

### 2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=1`.


```python
# Include relevant imports 
from sklearn.model_selection import train_test_split

# Create training and test sets with test_size=0.5 and random_state=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

### 3) Fit a decision tree regression model with scikit-learn to the training data. Use parameter defaults and `random_state=1` for this model. Then use the fitted regressor to generate predictions for the test data.

For the rest of this section feel free to refer to the scikit-learn documentation on [decision tree regressors](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).


```python
# Bring in necessary imports 
from sklearn.tree import DecisionTreeRegressor

# Fit the model to the training data 
dt = DecisionTreeRegressor(random_state=1)
dt.fit(X_train, y_train)

# Generate predictions for the test data
y_pred = dt.predict(X_test)
```

### 4) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set.

You can use the `sklearn.metrics` module.


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
```

    Mean Squared Error: 22.21041691053512
    Mean Absolute Error: 3.223405100334449
    R-squared: 0.9250580726905822


Hint: MSE should be about 22.21

### Hyperparameter Tuning of Decision Trees for Regression

### 5) Create a second decision tree model, this time with additional hyperparameters specified (still with `random_state`=1). Fit it to the training data, and generate predictions for the test data.


```python
# Evaluate the model on test data 
dt_tuned = DecisionTreeRegressor(
    random_state=1,
    max_depth=3,
    min_samples_leaf=2,
)
dt_tuned.fit(X_train,y_train)
y_pred_tuned = dt_tuned.predict(X_test)
```

### 6) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set from the new model. Did this improve your previous model? (It's ok if it didn't) Why or why not?


```python

print("Mean Squared Error:", mean_squared_error(y_test, y_pred_tuned))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_tuned))
print("R-squared:", r2_score(y_test, y_pred_tuned))
```

    Mean Squared Error: 26.619897534087755
    Mean Absolute Error: 4.01892182530331
    R-squared: 0.9101796947792777



```python

"""
Example: adjusting the max depth changes how many splits can happen on a single branch.
Setting this to three helped improve the model and reduced overfitting.
"""
```
