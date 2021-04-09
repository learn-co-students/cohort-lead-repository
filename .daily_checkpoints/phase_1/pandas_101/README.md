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
# Run this cell without changes
boston = load_boston()
```

The variable `boston` is now a dictionary with several key-value pairs containing different aspects of the Boston Housing dataset.  

#### What are the keys to `boston`?  


```python
# Your code here
```

#### Use the print command to print out the metadata for the dataset contained in the key `DESCR`


```python
# Your code here
```

#### Create a dataframe named "df_boston" with data contained in the key `data`.  Make the column names of `df_boston` the values from the key `feature_names`


```python
# Your code here
```

The key `target` contains the median value of a house.  

#### Add a column named "MEDV" to your dataframe which contains the median value of a house


```python
# Your code here
```

## Data Exploration

#### Show the first 5 rows of the dataframe with the `head` method


```python
# Your code here
```

#### Show the summary statistics of all columns with the `describe` method


```python
# Your code here
```

#### Check the datatypes of all columns, and see how many nulls are in each column, using the `info` method


```python
# Your code here
```

## Data Selection

#### Select all values from the column that contains the weighted distances to five Boston employment centres

*Hint: you printed out the information about what information variables contain in a cell above*


```python
# Your code here
```

#### Select rows 10-20 from the AGE, NOX, and MEDV columns


```python
# Your code here
```

#### Select all rows where NOX is greater than .7 and CRIM is greater than 8


```python
# Your code here
```

## Data Manipulation

#### Add a column to the dataframe called "MEDV*TAX" which is the product of MEDV and TAX


```python
# Your code here
```

#### What is the average median value of houses located on the Charles River?


```python
# Your code here
```

#### Write a sentence that answers the above question


```python
# Your written answer here
```
