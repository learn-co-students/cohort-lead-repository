# Principal Components Analysis

### Training a model with PCA-extracted features

In this challenge, you'll apply the unsupervised learning technique of Principal Components Analysis to the wine dataset. 

You'll use the principal components of the dataset as features in a machine learning model. You'll use the extracted features to train a vanilla Random Forest Classifier, and compare model performance to a model trained without PCA-extracted features. 

In the cell below, we import the data for you, and we split the data into training and test sets. 


```python
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
X, y = load_wine(return_X_y=True)

wine = load_wine()
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**1) Fit PCA to the training data** 

Call the PCA instance you'll create `wine_pca`. Set `n_components=0.9` and make sure to use `random_state = 42`.

_Hint: Make sure to include necessary imports for **preprocessing the data!**_


```python
# Relevant imports 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

# Create and fit an instance of PCA. Call it `wine_pca`. 
wine_pca = PCA(n_components = 0.9, random_state=42)
wine_pca.fit(X_train_scaled)
```

**2) How many principal components are there in the fitted PCA object?**

_Hint: Look at the list of attributes of trained `PCA` objects in the scikit-learn documentation_


```python
print(wine_pca.n_components_)
```

*Hint: you should end up with 8 components.*

Next, you'll reduce the dimensionality of the training data to the number of components that explain at least 90% of the variance in the data, and then you'll use this transformed data to fit a Random Forest classification model. 

You'll compare the performance of the model trained on the PCA-extracted features to the performance of a model trained using all features without feature extraction.

**3) Transform the training features into an array of reduced dimensionality using the `wine_pca` PCA object you've fit in the previous cell.**

Call this array `X_train_pca`.


```python
X_train_pca = wine_pca.transform(X_train_scaled)
```

Next, we create a dataframe from this array of transformed features and we inspect the first five rows of the dataframe for you. 


```python
# Create a dataframe from this array of transformed features 
X_train_pca = pd.DataFrame(X_train_pca)

# Inspect the first five rows of the transformed features dataset 
X_train_pca.head()
```

#### You will now use the PCA-extracted features to train a random forest classification model.

**4) Instantiate a Random Forest Classifier (call it `rfc`) and fit it to the transformed training data.**

Set `n_estimators=10`, `random_state=42`, and make sure you include the relevant import(s).


```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train_pca, y_train)
```

**5) Evaluate model performance on the test data and place model predictions in a variable called `y_pca_pred`.**

_Hint: Make sure to transform the test data the same way as you transformed the training data!!!_


```python
# Scale the test data using the `scaler` object 
X_test_scaled = scaler.transform(X_test)

# Transform the scaled test data using the `wine_pca` object
X_test_pca = wine_pca.transform(X_test_scaled)
X_test_pca = pd.DataFrame(X_test_pca)

# Evaluate model performance on transformed test data
y_pca_pred = rfc.predict(X_test_pca)
```

In the cell below, we print the classification report for the model performance on the test data. 


```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pca_pred))
```

Run the cell below to fit a vanilla Random Forest Classifier to the untransformed training data,  evaluate its performance on the untransformed test data, and print the classification report for the model. 


```python
vanilla_rfc = RandomForestClassifier(n_estimators=10, random_state=42)
vanilla_rfc.fit(X_train, y_train)

y_pred = vanilla_rfc.predict(X_test)

print(classification_report(y_test, y_pred))
```

**6) Compare model performance. Did the overall accuracy of the model improve when using the transformed features?**


```python
# The model accuracy for the model trained using the PCA-extracted features increased
# relative to the model trained using the untransformed features. 
```
