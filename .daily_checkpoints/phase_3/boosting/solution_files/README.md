## 1. What are two major differences between bagging and boosting?


```python

# One major distinction is that the learners in a bagging model are trained
# simultaneously and independently, while boosting trains learners iteratively.

# Relatedly, the bagged learners are strong learners, while boosting begins
# with a weak learner and aims to improve it.

# The final predictions for a bagged model come from a simple average over
# the individual learners, while, in a boosting model, the learners that
# get unusually correct predictions are accorded more weight.
```

## Gradient Boosting in Scikit-Learn

## 2. Fit a gradient-booster to `X_train_dums` with the following parameters:
- verbose=1
- learning_rate=0.2
- random_state=42


```python

boost = GradientBoostingClassifier(verbose=1, random_state=42, learning_rate=0.2)

boost.fit(X_train_dums, y_train)
```

## 3. What does the printout tell us?


```python

# The printout shows the iteration count, the training error for the learner at
# that iteration, and the time remaining for the training run. The training
# loss decreases at every iteration, which is a sign that the boosting is
# working.
```

## 4. Calculate and interpret the F-1 score for this model.


```python

from sklearn.metrics import f1_score
f1_score(y_test, boost.predict(X_test_dums))
```


```python

# The F-1 score is the harmonic mean of the precision and the recall.
# A score of 0.5 means that the sum of false positive and false negatives
# outnumber the true positives 2:1. So a score of 0.6 is pretty good.
```

## 5. What does the code below do? And what does it tell us about the model's predictions for Haitians?


```python

# The code first adds actual and predicted values (for salary > $50k) to the test data.
# Then it isolates the records from Haiti and compares the actual to the predicted values.
# The fact that `haiti['diff'].sum() / len(haiti)` = 1 means that the model correctly
# predicted the target for all Haitians.
```
