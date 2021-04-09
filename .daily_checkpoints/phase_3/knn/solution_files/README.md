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
k is the number of neighbors to consider
"""
```

### 2. How do the variance and bias of my model change as I adjust $k$? What would happen if I set $k$ to $n$, the size of my dataset?


```python
"""
In general, as k increases, model bias increases and model variance decreases.
As k decreases, model bias decreases and model variance increases.

If I were to set k to n, then the model would be totally biased in favor of
the most populous class and this would become the prediction for every data
point.
"""
```

## $k$-Nearest Neighbors in Scikit-Learn

In this section, you will fit a classification model to the wine dataset. The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different measurements taken for different constituents found in the three types of wine.


```python
wine = load_wine()
print(wine.DESCR)
```

    .. _wine_dataset:
    
    Wine recognition dataset
    ------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 178 (50 in each of three classes)
        :Number of Attributes: 13 numeric, predictive attributes and the class
        :Attribute Information:
     		- Alcohol
     		- Malic acid
     		- Ash
    		- Alcalinity of ash  
     		- Magnesium
    		- Total phenols
     		- Flavanoids
     		- Nonflavanoid phenols
     		- Proanthocyanins
    		- Color intensity
     		- Hue
     		- OD280/OD315 of diluted wines
     		- Proline
    
        - class:
                - class_0
                - class_1
                - class_2
    		
        :Summary Statistics:
        
        ============================= ==== ===== ======= =====
                                       Min   Max   Mean     SD
        ============================= ==== ===== ======= =====
        Alcohol:                      11.0  14.8    13.0   0.8
        Malic Acid:                   0.74  5.80    2.34  1.12
        Ash:                          1.36  3.23    2.36  0.27
        Alcalinity of Ash:            10.6  30.0    19.5   3.3
        Magnesium:                    70.0 162.0    99.7  14.3
        Total Phenols:                0.98  3.88    2.29  0.63
        Flavanoids:                   0.34  5.08    2.03  1.00
        Nonflavanoid Phenols:         0.13  0.66    0.36  0.12
        Proanthocyanins:              0.41  3.58    1.59  0.57
        Colour Intensity:              1.3  13.0     5.1   2.3
        Hue:                          0.48  1.71    0.96  0.23
        OD280/OD315 of diluted wines: 1.27  4.00    2.61  0.71
        Proline:                       278  1680     746   315
        ============================= ==== ===== ======= =====
    
        :Missing Attribute Values: None
        :Class Distribution: class_0 (59), class_1 (71), class_2 (48)
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML Wine recognition datasets.
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    
    The data is the results of a chemical analysis of wines grown in the same
    region in Italy by three different cultivators. There are thirteen different
    measurements taken for different constituents found in the three types of
    wine.
    
    Original Owners: 
    
    Forina, M. et al, PARVUS - 
    An Extendible Package for Data Exploration, Classification and Correlation. 
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.
    
    Citation:
    
    Lichman, M. (2013). UCI Machine Learning Repository
    [https://archive.ics.uci.edu/ml]. Irvine, CA: University of California,
    School of Information and Computer Science. 
    
    .. topic:: References
    
      (1) S. Aeberhard, D. Coomans and O. de Vel, 
      Comparison of Classifiers in High Dimensional Settings, 
      Tech. Rep. no. 92-02, (1992), Dept. of Computer Science and Dept. of  
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Technometrics). 
    
      The data was used with many others for comparing various 
      classifiers. The classes are separable, though only RDA 
      has achieved 100% correct classification. 
      (RDA : 100%, QDA 99.4%, LDA 98.9%, 1NN 96.1% (z-transformed data)) 
      (All results using the leave-one-out technique) 
    
      (2) S. Aeberhard, D. Coomans and O. de Vel, 
      "THE CLASSIFICATION PERFORMANCE OF RDA" 
      Tech. Rep. no. 92-01, (1992), Dept. of Computer Science and Dept. of 
      Mathematics and Statistics, James Cook University of North Queensland. 
      (Also submitted to Journal of Chemometrics).
    



```python
wine.feature_names
```




    ['alcohol',
     'malic_acid',
     'ash',
     'alcalinity_of_ash',
     'magnesium',
     'total_phenols',
     'flavanoids',
     'nonflavanoid_phenols',
     'proanthocyanins',
     'color_intensity',
     'hue',
     'od280/od315_of_diluted_wines',
     'proline']




```python
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df.head()
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
      <th>alcohol</th>
      <th>malic_acid</th>
      <th>ash</th>
      <th>alcalinity_of_ash</th>
      <th>magnesium</th>
      <th>total_phenols</th>
      <th>flavanoids</th>
      <th>nonflavanoid_phenols</th>
      <th>proanthocyanins</th>
      <th>color_intensity</th>
      <th>hue</th>
      <th>od280/od315_of_diluted_wines</th>
      <th>proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127.0</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100.0</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101.0</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113.0</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118.0</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
wine.target[:5]
```




    array([0, 0, 0, 0, 0])



### 3. Perform a train-test split with `random_state=6`, scale, and then fit a $k$-Nearest Neighbors Classifier to the training data with $k$ = 7.


```python

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_sc, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                         weights='uniform')



### Confusion Matrix


```python
confusion_matrix(y_test, knn.predict(X_test_sc))
```




    array([[15,  0,  0],
           [ 1, 16,  0],
           [ 0,  0, 13]])



### 4. How accurate is the model?  What is the precision of the model in classifying wines from *Class 0*?  What is the recall of the model in classifying wines from *Class 1*?


```python

# To calculate accuracy, we can
# use knn.score(); or
print(knn.score(X_test_sc, y_test))
# import accuracy_score() from sklearn.metrics; or
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, knn.predict(X_test_sc)))
# add the entries on the main diagonal and divide by number of test wines:
print((15 + 16 + 13) / len(y_test))
```

    0.9777777777777777
    0.9777777777777777
    0.9777777777777777



```python

# To calculate precision, we can
# import precision_score() from sklearn.metrics; or
from sklearn.metrics import precision_score
print(precision_score(y_test, knn.predict(X_test_sc), average=None)[0])
# divide the number of true positives by the sum of true and false positives:
print(15 / (15 + 1 + 0))
```

    0.9375
    0.9375



```python
# To calculate recall, we can
# import recall_score() from sklearn.metrics; or
from sklearn.metrics import recall_score
print(recall_score(y_test, knn.predict(X_test_sc), average=None)[1])
# divide the number of true positives by the sum of true positives and false negatives:
print(16 / (1 + 16 + 0))
```

    0.9411764705882353
    0.9411764705882353



```python
"""
This model has about:
- 98% accuracy
- 94% precision in classifying wines from Class 0
- 94% recall in classifying wines from Class 1
"""
```

### Now try a model with $k$ = 5 and a Manhattan distance metric. (You can use the same train-test split.)


```python

knn2 = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

knn2.fit(X_train_sc, y_train)

confusion_matrix(y_test, knn2.predict(X_test_sc))
```




    array([[15,  0,  0],
           [ 0, 16,  1],
           [ 0,  0, 13]])



### 5. How accurate is the new model? What is the precision of the model in classifying wines from *Class 0*?  What is the recall of the model in classifying wines from *Class 1*?  Which model is better? (We may or may not have enough information to make this determination)


```python
print(knn2.score(X_test_sc, y_test))
print(accuracy_score(y_test, knn2.predict(X_test_sc)))
print((15 + 16 + 13) / len(y_test))
```

    0.9777777777777777
    0.9777777777777777
    0.9777777777777777



```python
print(precision_score(y_test, knn2.predict(X_test_sc), average=None)[0])
print(15 / (15 + 0 + 0))
```

    1.0
    1.0



```python
print(recall_score(y_test, knn2.predict(X_test_sc), average=None)[1])
print(16 / (0 + 16 + 1))
```

    0.9411764705882353
    0.9411764705882353



```python
"""
The new model has:
- 98% accuracy
- 100% precision in classifying wines from Class 0
- 94% recall in classifying wines from Class 1

In comparison to the previous model, the new model has the same accuracy
and recall in classifying wines from Class 1, and better precision in
classifying wines from Class 0.  In general this means we can assume the
second model is better, if these are the metrics that matter to us.

A stronger answer would also include the fact that this is a very small
set of data, so the difference between these models is just the difference
of 1 wine being incorrectly categorized as Class 0 and 1 wine being
incorrectly categorized as Class 2, so in fact the first or the second
might be better if more data becomes available
"""
```
