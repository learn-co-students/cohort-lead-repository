# Neural Network Regularization

This assessment covers building and training a `tf.keras` `Sequential` model, then applying regularization.  The dataset comes from a ["don't overfit" Kaggle competition](https://www.kaggle.com/c/dont-overfit-ii).  There are 300 features labeled 0-299, and a target called "target".  There are only 250 records total, meaning this is a very small dataset to be used with a neural network. 

_You can assume that the dataset has already been scaled._


```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
tf.logging.set_verbosity(tf.logging.ERROR)
```

In the cells below, the set of data has been split into a training and testing set and then fit to a neural network with two hidden layers. Run the cells below to see how well the model performs.


```python
df = pd.read_csv("data.csv")
df.drop("id", axis=1, inplace=True)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
X_train.shape
```




    (187, 300)




```python

def build_model():
    classifier = Sequential()
    classifier.add(Dense(units=64, input_shape=(300,)))
    classifier.add(Dense(units=64))
    classifier.add(Dense(units=64))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    return classifier
```


```python
def fit_and_cross_validate_model(model_func, X, y):
    """
    Given a function that builds a model and training X and y, validate the model based on
    cross-validated train and test data
    """
    keras_classifier = KerasClassifier(build_model, epochs=5, batch_size=50, verbose=1, shuffle=False)
    print("######################## Training cross-validated models ###########################")
    cross_val_scores = cross_val_score(keras_classifier, X, y, cv=5)
    print("########################### Training on full X_train ###############################")
    keras_classifier.fit(X, y)
    print("############################### Evaluation report ##################################")
    print("Approximate training accuracy:")
    print(accuracy_score(y, keras_classifier.predict(X)))
    print("Approximate testing accuracy:")
    print(np.mean(cross_val_scores), "+/-", np.std(cross_val_scores))
```


```python
fit_and_cross_validate_model(build_model, X_train, y_train);
```

    ######################## Training cross-validated models ###########################
    Epoch 1/5
    149/149 [==============================] - 0s 2ms/sample - loss: 1.0130 - acc: 0.4698
    Epoch 2/5
    149/149 [==============================] - 0s 109us/sample - loss: 0.6362 - acc: 0.6242
    Epoch 3/5
    149/149 [==============================] - 0s 124us/sample - loss: 0.4664 - acc: 0.7987
    Epoch 4/5
    149/149 [==============================] - 0s 91us/sample - loss: 0.3688 - acc: 0.8523
    Epoch 5/5
    149/149 [==============================] - 0s 130us/sample - loss: 0.2980 - acc: 0.8792
    38/38 [==============================] - 0s 2ms/sample - loss: 0.8184 - acc: 0.5789
    Epoch 1/5
    149/149 [==============================] - 0s 2ms/sample - loss: 0.8932 - acc: 0.4899
    Epoch 2/5
    149/149 [==============================] - 0s 133us/sample - loss: 0.5847 - acc: 0.6913
    Epoch 3/5
    149/149 [==============================] - 0s 127us/sample - loss: 0.4323 - acc: 0.8322
    Epoch 4/5
    149/149 [==============================] - 0s 111us/sample - loss: 0.3329 - acc: 0.8859
    Epoch 5/5
    149/149 [==============================] - 0s 88us/sample - loss: 0.2548 - acc: 0.9329
    38/38 [==============================] - 0s 2ms/sample - loss: 1.0012 - acc: 0.6579
    Epoch 1/5
    150/150 [==============================] - 0s 2ms/sample - loss: 1.0751 - acc: 0.4800
    Epoch 2/5
    150/150 [==============================] - 0s 105us/sample - loss: 0.6667 - acc: 0.6467
    Epoch 3/5
    150/150 [==============================] - 0s 122us/sample - loss: 0.4763 - acc: 0.7733
    Epoch 4/5
    150/150 [==============================] - 0s 104us/sample - loss: 0.3695 - acc: 0.8800
    Epoch 5/5
    150/150 [==============================] - 0s 78us/sample - loss: 0.2930 - acc: 0.9000
    37/37 [==============================] - 0s 2ms/sample - loss: 0.8841 - acc: 0.5946
    Epoch 1/5
    150/150 [==============================] - 0s 2ms/sample - loss: 0.8584 - acc: 0.5600
    Epoch 2/5
    150/150 [==============================] - 0s 108us/sample - loss: 0.5741 - acc: 0.7133
    Epoch 3/5
    150/150 [==============================] - 0s 121us/sample - loss: 0.4419 - acc: 0.8333
    Epoch 4/5
    150/150 [==============================] - 0s 154us/sample - loss: 0.3571 - acc: 0.9133
    Epoch 5/5
    150/150 [==============================] - 0s 152us/sample - loss: 0.2861 - acc: 0.9333
    37/37 [==============================] - 0s 3ms/sample - loss: 0.6768 - acc: 0.6757
    Epoch 1/5
    150/150 [==============================] - 0s 2ms/sample - loss: 1.1179 - acc: 0.4467
    Epoch 2/5
    150/150 [==============================] - 0s 127us/sample - loss: 0.6902 - acc: 0.6000
    Epoch 3/5
    150/150 [==============================] - 0s 172us/sample - loss: 0.4900 - acc: 0.7200
    Epoch 4/5
    150/150 [==============================] - 0s 154us/sample - loss: 0.3694 - acc: 0.8333
    Epoch 5/5
    150/150 [==============================] - 0s 133us/sample - loss: 0.2806 - acc: 0.9333
    37/37 [==============================] - 0s 4ms/sample - loss: 0.9008 - acc: 0.5135
    ########################### Training on full X_train ###############################
    Epoch 1/5
    187/187 [==============================] - 0s 2ms/sample - loss: 1.1288 - acc: 0.4545
    Epoch 2/5
    187/187 [==============================] - 0s 113us/sample - loss: 0.6296 - acc: 0.6417
    Epoch 3/5
    187/187 [==============================] - 0s 98us/sample - loss: 0.4577 - acc: 0.8128
    Epoch 4/5
    187/187 [==============================] - 0s 202us/sample - loss: 0.3693 - acc: 0.8610
    Epoch 5/5
    187/187 [==============================] - 0s 108us/sample - loss: 0.2969 - acc: 0.8877
    ############################### Evaluation report ##################################
    Approximate training accuracy:
    187/187 [==============================] - 0s 456us/sample
    0.9411764705882353
    Approximate testing accuracy:
    0.60412517786026 +/- 0.05821661272907996


## 1) Modify the code below to use regularization


The model appears to be overfitting. To deal with this overfitting, modify the code below to include regularization in the model. You can add L1, L2, both L1 and L2, or dropout regularization.

Hint: these might be helpful

 - [`Dense` layer documentation](https://keras.io/layers/core/)
 - [`regularizers` documentation](https://keras.io/regularizers/)


```python
def build_model_with_regularization():
    classifier = Sequential()
    classifier.add(Dense(units=64, input_shape=(300,), kernel_regularizer=regularizers.l2(0.0000000000000001)))
    # they might add a kernel regularizer
    classifier.add(Dense(units=64, kernel_regularizer=regularizers.l2(0.0000000000000001)))
    # they might add a dropout layer
    classifier.add(Dropout(0.8))
    classifier.add(Dense(units=64, kernel_regularizer=regularizers.l2(0.0000000000000001)))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

```


```python
fit_and_cross_validate_model(build_model_with_regularization, X_train, y_train);
```

    ######################## Training cross-validated models ###########################
    Epoch 1/5
    149/149 [==============================] - 0s 2ms/sample - loss: 0.8535 - acc: 0.5302
    Epoch 2/5
    149/149 [==============================] - 0s 121us/sample - loss: 0.5429 - acc: 0.7248
    Epoch 3/5
    149/149 [==============================] - 0s 327us/sample - loss: 0.3991 - acc: 0.8456
    Epoch 4/5
    149/149 [==============================] - 0s 262us/sample - loss: 0.3093 - acc: 0.8658
    Epoch 5/5
    149/149 [==============================] - 0s 122us/sample - loss: 0.2406 - acc: 0.9128
    38/38 [==============================] - 0s 4ms/sample - loss: 0.8746 - acc: 0.5789
    Epoch 1/5
    149/149 [==============================] - 0s 3ms/sample - loss: 0.9684 - acc: 0.5101
    Epoch 2/5
    149/149 [==============================] - 0s 156us/sample - loss: 0.5478 - acc: 0.6913
    Epoch 3/5
    149/149 [==============================] - 0s 128us/sample - loss: 0.3774 - acc: 0.8523
    Epoch 4/5
    149/149 [==============================] - 0s 139us/sample - loss: 0.2825 - acc: 0.8993
    Epoch 5/5
    149/149 [==============================] - 0s 163us/sample - loss: 0.2096 - acc: 0.9463
    38/38 [==============================] - 0s 4ms/sample - loss: 0.9508 - acc: 0.5526
    Epoch 1/5
    150/150 [==============================] - 0s 3ms/sample - loss: 0.7792 - acc: 0.5600
    Epoch 2/5
    150/150 [==============================] - 0s 129us/sample - loss: 0.5105 - acc: 0.7400
    Epoch 3/5
    150/150 [==============================] - 0s 120us/sample - loss: 0.3802 - acc: 0.8533
    Epoch 4/5
    150/150 [==============================] - 0s 166us/sample - loss: 0.2944 - acc: 0.9200
    Epoch 5/5
    150/150 [==============================] - 0s 115us/sample - loss: 0.2266 - acc: 0.9467
    37/37 [==============================] - 0s 4ms/sample - loss: 1.0143 - acc: 0.5135
    Epoch 1/5
    150/150 [==============================] - 0s 3ms/sample - loss: 0.9874 - acc: 0.4533
    Epoch 2/5
    150/150 [==============================] - 0s 96us/sample - loss: 0.6195 - acc: 0.7000
    Epoch 3/5
    150/150 [==============================] - 0s 117us/sample - loss: 0.4555 - acc: 0.7933
    Epoch 4/5
    150/150 [==============================] - 0s 124us/sample - loss: 0.3512 - acc: 0.8667
    Epoch 5/5
    150/150 [==============================] - 0s 97us/sample - loss: 0.2652 - acc: 0.9533
    37/37 [==============================] - 0s 5ms/sample - loss: 0.7550 - acc: 0.6216
    Epoch 1/5
    150/150 [==============================] - 1s 4ms/sample - loss: 1.0522 - acc: 0.5067
    Epoch 2/5
    150/150 [==============================] - 0s 158us/sample - loss: 0.6428 - acc: 0.6800
    Epoch 3/5
    150/150 [==============================] - 0s 231us/sample - loss: 0.4589 - acc: 0.7933
    Epoch 4/5
    150/150 [==============================] - 0s 100us/sample - loss: 0.3493 - acc: 0.8800
    Epoch 5/5
    150/150 [==============================] - 0s 119us/sample - loss: 0.2693 - acc: 0.9400
    37/37 [==============================] - 0s 6ms/sample - loss: 0.9054 - acc: 0.5135
    ########################### Training on full X_train ###############################
    Epoch 1/5
    187/187 [==============================] - 0s 3ms/sample - loss: 0.9576 - acc: 0.4599
    Epoch 2/5
    187/187 [==============================] - 0s 153us/sample - loss: 0.6224 - acc: 0.6417
    Epoch 3/5
    187/187 [==============================] - 0s 172us/sample - loss: 0.4819 - acc: 0.8021
    Epoch 4/5
    187/187 [==============================] - 0s 259us/sample - loss: 0.3878 - acc: 0.8556
    Epoch 5/5
    187/187 [==============================] - 0s 107us/sample - loss: 0.3064 - acc: 0.8930
    ############################### Evaluation report ##################################
    Approximate training accuracy:
    187/187 [==============================] - 0s 913us/sample
    0.9144385026737968
    Approximate testing accuracy:
    0.5560455083847046 +/- 0.041120110886002884


### Based on the cross-validated scores, did the regularization you performed help prevent overfitting? Is the first or the second model better?


```python
# It may or may not have prevented overfitting, depending on random elements
# within the neural net as well as their choice of regularization technique
#
# (TensorFlow + random seeding is not fully possible in a Jupyter Notebook)
#
# The student should interpret the numbers they have
#
# In the example given above, a reasonable answer would be:
# The regularization is helping to prevent overfitting, but it also might be
# causing some underfitting.  The train and test accuracy are more similar to
# each other, but the test accuracy also got slightly worse.  I think the
# original model is better, even though it is overfitting.
#
# It is also very likely that they will not have applied strong enough
# regularization to make a difference, so the scores for the two models will
# mainly differ based on random seeds
```

### Now, evaluate both models on the holdout set


```python
classifier_1 = build_model()
classifier_1.fit(X_train, y_train, epochs=5, verbose=1, batch_size=50, shuffle=False)

classifier_2 = build_model_with_regularization()
classifier_2.fit(X_train, y_train, epochs=5, verbose=1, batch_size=50, shuffle=False)

print("Accuracy score without regularization:", accuracy_score(y_test, classifier_1.predict_classes(X_test)))
print("Accuracy score with regularization:", accuracy_score(y_test, classifier_2.predict_classes(X_test)))
```

    Epoch 1/5
    187/187 [==============================] - 0s 2ms/sample - loss: 1.0741 - acc: 0.4652
    Epoch 2/5
    187/187 [==============================] - 0s 145us/sample - loss: 0.6118 - acc: 0.7112
    Epoch 3/5
    187/187 [==============================] - 0s 93us/sample - loss: 0.4258 - acc: 0.8449
    Epoch 4/5
    187/187 [==============================] - 0s 105us/sample - loss: 0.3201 - acc: 0.9037
    Epoch 5/5
    187/187 [==============================] - 0s 142us/sample - loss: 0.2375 - acc: 0.9412
    Epoch 1/5
    187/187 [==============================] - 1s 3ms/sample - loss: 1.9396 - acc: 0.4492
    Epoch 2/5
    187/187 [==============================] - 0s 177us/sample - loss: 1.4191 - acc: 0.5080
    Epoch 3/5
    187/187 [==============================] - 0s 167us/sample - loss: 1.1469 - acc: 0.5455
    Epoch 4/5
    187/187 [==============================] - 0s 128us/sample - loss: 0.9037 - acc: 0.6417
    Epoch 5/5
    187/187 [==============================] - 0s 86us/sample - loss: 0.8540 - acc: 0.5775
    Accuracy score without regularization: 0.6031746031746031
    Accuracy score with regularization: 0.5396825396825397


### 2) Explain how regularization is related to the bias/variance tradeoff within Neural Networks and how it's related to the results you just achieved in the training and test accuracies of the previous models. What does regularization change in the training process (be specific to what is being regularized and how it is regularizing)?



```python
# Regularization helps prevent over fitting by adding penalty terms to the cost function. 
# This prevents any one feature to having too much importance in a model.  One feature
# having too much importance can lead to overfitting (high variance).  On the other hand,
# too much regularization can lead to underfitting (high bias).
#
# The specific regularization used in the solution code is:
# L2 regularization: penalizes weight matrices for being too large
# Dropout regularization: a random subset of nodes are ignored
#
# The current dataset is very small to be used with a neural network, so it's possible that
# we don't actually have enough information to create a good, generalizable model
```

### 3) How might L1  and dropout regularization change a neural network's architecture?


```python
# L1 and dropout regularization may eliminate connections between nodes entirely.
```
