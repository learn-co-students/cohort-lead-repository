# Ensemble Methods

### Introduction to Ensemble Methods

**1) Explain how the random forest algorithm works. Why are random forests resilient to overfitting?**

_Hint: Your answer should discuss bagging and the subspace sampling method._


```python
# Your written answer here
```

### Random Forests and Hyperparameter Tuning using GridSearchCV

In this section, you will perform hyperparameter tuning for a Random Forest classifier using GridSearchCV. You will use `scikit-learn`'s wine dataset to classify wines into one of three different classes. 

After finding the best estimator, you will interpret the best model's feature importances. 

In the cells below, we have loaded the relevant imports and the wine data for you. 


```python
# Relevant imports
import pandas as pd 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```

In the cells below, we inspect the first five rows of the dataframe and compute the dataframe's shape.


```python
df.head()
```


```python
df.shape
```

We also get descriptive statistics for the dataset features, and obtain the distribution of classes in the dataset. 


```python
X.describe()
```


```python
y.value_counts().sort_index()
```

You will now perform hyper-parameter tuning for a Random Forest classifier.

**2) Construct a `param_grid` dictionary to pass to `GridSearchCV` when instantiating the object. Choose at least 3 hyper-parameters to tune and 3 values for each.** 


```python
# Replace None with relevant code 
param_grid = None
```

Now that you have created the `param_grid` dictionary of hyperparameters, let's continue performing hyperparameter optimization of a Random Forest Classifier. 

In the cell below, we include the relevant imports for you.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

**3) Create an instance of a Random Forest Classifier estimator; call it `rfc`.** Make sure to set `random_state=42` for reproducibility. 


```python
# Replace None with appropriate code
rfc = None
```

**4) Create an instance of an `GridSearchCV` object and fit it to the data.** Call the instance `cv_rfc`. 

* Use the random forest classification estimator you instantiated in the cell above, the parameter grid dictionary constructed, and make sure to perform 5-fold cross validation. 
* The fitting process should take 10 - 15 seconds to complete. 


```python
# Replace None with appropriate code 
cv_rfc = None 

cv_rfc.fit(None, None)
```

**5) What are the best training parameters found by GridSearchCV?** 

_Hint: Explore the documentation for GridSearchCV._ 


```python
# Your code here
```

In the cell below, we create a variable `best_model` that holds the best model found by the grid search.


```python
best_model = cv_rfc.best_estimator_
```

Next, we give you a function that creates a horizontal bar plot to visualize the feature importances of a model, sorted in descending order. 


```python
import matplotlib.pyplot as plt 
%matplotlib inline 

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 
    
    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''
    
    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances, 
                                     key = lambda x: x[1], reverse=True)
    
    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('importance')
```

**6) Create a plot of the best model's feature importances.** 

_Hint: To create the plot, pass the appropriate parameters to the function above._


```python
# Your code here
```

**7) What are this model's top 3 features in order of descending importance?**


```python
# Your written answer here 
```
