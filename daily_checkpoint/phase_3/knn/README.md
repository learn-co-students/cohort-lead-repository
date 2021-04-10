# k-Nearest Neighbors


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
```

### 1. What does the '$k$' represent in "$k$-Nearest Neighbors"?


```python
"""
Your written answer here
"""
```

### 2. How do the variance and bias of my model change as I adjust $k$? What would happen if I set $k$ to $n$, the size of my dataset?


```python
"""
Your written answer here
"""
```

## $k$-Nearest Neighbors in Scikit-Learn

In this section, you will fit a classification model to the wine dataset. The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different measurements taken for different constituents found in the three types of wine.


```python
# Run this cell without changes
wine = load_wine()
print(wine.DESCR)
```


```python
# Run this cell without changes
wine.feature_names
```


```python
# Run this cell without changes
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df.head()
```


```python
# Run this cell without changes
wine.target[:5]
```

### 3. Perform a train-test split with `random_state=6`, scale, and then fit a $k$-Nearest Neighbors Classifier to the training data with $k$ = 7.


```python
# Replace None with appropriate code

# Perform a train-test split
X = None
y = None

X_train, X_test, y_train, y_test = None

# Scale
ss = None
X_train_sc = None
X_test_sc = None

# Create and fit a k-Nearest Neighbors Classifier
knn = None
knn.fit(None, None)
```

### Confusion Matrix


```python
# Run this cell without changes
confusion_matrix(y_test, knn.predict(X_test_sc))
```

### 4. How accurate is the model?  What is the precision of the model in classifying wines from *Class 0*?  What is the recall of the model in classifying wines from *Class 1*?


```python
# Your code here to calculate accuracy

```


```python
# Your code here to calculate precision for Class 0

```


```python
# Your code here to calculate recall for Class 1

```


```python
"""
Your written answer here
"""
```

### Now try a model with $k$ = 5 and a Manhattan distance metric. (You can use the same train-test split.)


```python
# Your code here


```

### 5. How accurate is the new model? What is the precision of the model in classifying wines from *Class 0*?  What is the recall of the model in classifying wines from *Class 1*?  Which model is better? (We may or may not have enough information to make this determination)


```python
# Your code here to calculate accuracy

```


```python
# Your code here to calculate precision for Class 0

```


```python
# Your code here to calculate recall for Class 1

```


```python
"""
Your written answer here
"""
```
