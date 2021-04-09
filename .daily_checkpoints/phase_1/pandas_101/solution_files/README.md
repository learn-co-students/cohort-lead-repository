# Pandas 101

This checkpoint contains many of the basic tasks you might need to do with Pandas!  At the end of an hour, commit and push what you have (remember, you can always return to this book later for practice)


```python
# Run this import cell without changes

#data manipulation
import pandas as pd

#dataset
from sklearn.datasets import load_boston
```

## Loading in the Boston Housing Dataset


```python
boston = load_boston()
```

The variable `boston` is now a dictionary with several key-value pairs containing different aspects of the Boston Housing dataset.  

#### What are the keys to `boston`?  


```python
boston.keys()
```




    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])



#### Use the print command to print out the metadata for the dataset contained in the key `DESCR`


```python
print(boston['DESCR'])
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    


#### Create a dataframe named "df_boston" with data contained in the key `data`.  Make the column names of `df_boston` the values from the key `feature_names`


```python
    
df_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
```

The key `target` contains the median value of a house.  

#### Add a column named "MEDV" to your dataframe which contains the median value of a house


```python

df_boston['MEDV'] = boston['target']
```

## Data Exploration

#### Show the first 5 rows of the dataframe with the `head` method


```python
df_boston.head()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Show the summary statistics of all columns with the `describe` method


```python
df_boston.describe()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Check the datatypes of all columns, and see how many nulls are in each column, using the `info` method


```python

df_boston.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    float64
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    float64
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  MEDV     506 non-null    float64
    dtypes: float64(14)
    memory usage: 55.5 KB


## Data Selection

#### Select all values from the column that contains the weighted distances to five Boston employment centres

*Hint: you printed out the information about what information variables contain in a cell above*


```python
df_boston['DIS']
```




    0      4.0900
    1      4.9671
    2      4.9671
    3      6.0622
    4      6.0622
            ...  
    501    2.4786
    502    2.2875
    503    2.1675
    504    2.3889
    505    2.5050
    Name: DIS, Length: 506, dtype: float64



#### Select rows 10-20 from the AGE, NOX, and MEDV columns


```python
df_boston.loc[10:20, ['AGE', 'NOX', 'MEDV']]
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
      <th>AGE</th>
      <th>NOX</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>94.3</td>
      <td>0.524</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>82.9</td>
      <td>0.524</td>
      <td>18.9</td>
    </tr>
    <tr>
      <th>12</th>
      <td>39.0</td>
      <td>0.524</td>
      <td>21.7</td>
    </tr>
    <tr>
      <th>13</th>
      <td>61.8</td>
      <td>0.538</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>84.5</td>
      <td>0.538</td>
      <td>18.2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>56.5</td>
      <td>0.538</td>
      <td>19.9</td>
    </tr>
    <tr>
      <th>16</th>
      <td>29.3</td>
      <td>0.538</td>
      <td>23.1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>81.7</td>
      <td>0.538</td>
      <td>17.5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>36.6</td>
      <td>0.538</td>
      <td>20.2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>69.5</td>
      <td>0.538</td>
      <td>18.2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>98.1</td>
      <td>0.538</td>
      <td>13.6</td>
    </tr>
  </tbody>
</table>
</div>



#### Select all rows where NOX is greater than .7 and CRIM is greater than 8


```python
mask = (
    (df_boston['NOX']>.7) &
    (df_boston['CRIM']>8)
)
df_boston[mask]
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>356</th>
      <td>8.98296</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>1.0</td>
      <td>0.770</td>
      <td>6.212</td>
      <td>97.4</td>
      <td>2.1222</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>377.73</td>
      <td>17.60</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>419</th>
      <td>11.81230</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.718</td>
      <td>6.824</td>
      <td>76.5</td>
      <td>1.7940</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>48.45</td>
      <td>22.74</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>420</th>
      <td>11.08740</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.718</td>
      <td>6.411</td>
      <td>100.0</td>
      <td>1.8589</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>318.75</td>
      <td>15.02</td>
      <td>16.7</td>
    </tr>
    <tr>
      <th>434</th>
      <td>13.91340</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.208</td>
      <td>95.0</td>
      <td>2.2222</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>100.63</td>
      <td>15.17</td>
      <td>11.7</td>
    </tr>
    <tr>
      <th>435</th>
      <td>11.16040</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.629</td>
      <td>94.6</td>
      <td>2.1247</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>109.85</td>
      <td>23.27</td>
      <td>13.4</td>
    </tr>
    <tr>
      <th>436</th>
      <td>14.42080</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.461</td>
      <td>93.3</td>
      <td>2.0026</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>27.49</td>
      <td>18.05</td>
      <td>9.6</td>
    </tr>
    <tr>
      <th>437</th>
      <td>15.17720</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.152</td>
      <td>100.0</td>
      <td>1.9142</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>9.32</td>
      <td>26.45</td>
      <td>8.7</td>
    </tr>
    <tr>
      <th>438</th>
      <td>13.67810</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.935</td>
      <td>87.9</td>
      <td>1.8206</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>68.95</td>
      <td>34.02</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>439</th>
      <td>9.39063</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.627</td>
      <td>93.9</td>
      <td>1.8172</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>22.88</td>
      <td>12.8</td>
    </tr>
    <tr>
      <th>440</th>
      <td>22.05110</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.818</td>
      <td>92.4</td>
      <td>1.8662</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>391.45</td>
      <td>22.11</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>441</th>
      <td>9.72418</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.406</td>
      <td>97.2</td>
      <td>2.0651</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>385.96</td>
      <td>19.52</td>
      <td>17.1</td>
    </tr>
    <tr>
      <th>443</th>
      <td>9.96654</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.485</td>
      <td>100.0</td>
      <td>1.9784</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>386.73</td>
      <td>18.85</td>
      <td>15.4</td>
    </tr>
    <tr>
      <th>444</th>
      <td>12.80230</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>5.854</td>
      <td>96.6</td>
      <td>1.8956</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>240.52</td>
      <td>23.79</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>445</th>
      <td>10.67180</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.459</td>
      <td>94.8</td>
      <td>1.9879</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>43.06</td>
      <td>23.98</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>447</th>
      <td>9.92485</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.740</td>
      <td>6.251</td>
      <td>96.6</td>
      <td>2.1980</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>388.52</td>
      <td>16.44</td>
      <td>12.6</td>
    </tr>
    <tr>
      <th>448</th>
      <td>9.32909</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.185</td>
      <td>98.7</td>
      <td>2.2616</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>18.13</td>
      <td>14.1</td>
    </tr>
    <tr>
      <th>453</th>
      <td>8.24809</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>7.393</td>
      <td>99.3</td>
      <td>2.4527</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>375.87</td>
      <td>16.74</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>454</th>
      <td>9.51363</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.728</td>
      <td>94.1</td>
      <td>2.4961</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>6.68</td>
      <td>18.71</td>
      <td>14.9</td>
    </tr>
    <tr>
      <th>457</th>
      <td>8.20058</td>
      <td>0.0</td>
      <td>18.1</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>5.936</td>
      <td>80.3</td>
      <td>2.7792</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>3.50</td>
      <td>16.94</td>
      <td>13.5</td>
    </tr>
  </tbody>
</table>
</div>



## Data Manipulation

#### Add a column to the dataframe called "MEDV*TAX" which is the product of MEDV and TAX


```python
df_boston['MEDV*TAX'] = df_boston['MEDV']*df_boston['TAX']
```

#### What is the average median value of houses located on the Charles River?


```python

val = (
    df_boston
    [df_boston['CHAS']==1]
    ['MEDV']
    .mean()
)

val = val*1000
val
```




    28439.999999999996



#### Write a sentence that answers the above question


```python


'''The average median value of houses located along the Charles River is $28,440'''
```
