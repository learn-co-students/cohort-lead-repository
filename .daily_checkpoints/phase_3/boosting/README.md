## 1. What are two major differences between bagging and boosting?


```python
# Your answer here


```

## Gradient Boosting in Scikit-Learn


```python
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
```


```python
salaries = pd.read_csv('adult.data', header=None)
```


```python
salaries.head()
```


```python
salaries[14] = salaries[14].map(lambda x: False if x == ' <=50K' else True)
```


```python
X_train, X_test, y_train, y_test = train_test_split(salaries.drop(14, axis=1), salaries[14],
                                                   random_state=2)
```


```python
X_train
```


```python
ct = ColumnTransformer([("ohe", OneHotEncoder(handle_unknown='ignore'), [1, 3, 5, 6, 7, 8, 9, 13])])
```


```python
ct.fit_transform(X_train)
```


```python
X_train_dums = ct.transform(X_train)
X_test_dums = ct.transform(X_test)
```


```python
dummies = pd.DataFrame(X_train_dums.todense(), columns=ct.get_feature_names())
dummies.head()
```

## 2. Fit a gradient-booster to `X_train_dums` with the following parameters:
- verbose=1
- learning_rate=0.2
- random_state=42


```python
# Your answer here


```

## 3. What does the printout tell us?


```python
# Your answer here


```

## 4. Calculate and interpret the F-1 score for this model.


```python
# Your answer here


```

## 5. What does the code below do? And what does it tell us about the model's predictions for Haitians?


```python
test_df = pd.DataFrame(X_test_dums.todense(), columns=ct.get_feature_names(), index=X_test.index)
test_df['pred'] = boost.predict(X_test_dums)
test_df['actual'] = y_test



haiti = test_df[test_df['ohe__x7_ Haiti'] == 1].copy()

haiti['diff'] = haiti['actual'] == haiti['pred']
haiti['diff'].sum() / len(haiti)
```


```python
# Your answer here


```
