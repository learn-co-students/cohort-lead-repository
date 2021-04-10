## Pandas Data Cleaning

In this checkpoint you will be doing some preprocessing for a dataset for the videogame FIFA19 (https://www.kaggle.com/karangadiya/fifa19).  The dataset contains both data for the game as well as information about the players' real life careers.

**1) Read the CSV file into a pandas dataframe**

The data you'll be working with is found in a file called `'./data/fifa.csv'`.  Use your knowledge of pandas to create a new dataframe using the csv data. 

Check the contents of the first few rows of your dataframe, then show the size of the dataframe


```python
# Run this cell without changes to import the necessary libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```


```python
# Replace None with appropriate code
df = None
```


```python
# Code here to check the first few rows of the DataFrame
```


```python
# Code here to see the size of the DataFrame

```

**2. Drop rows with missing values for for `Release Clause`**
    
**Drop rows for which `Release Clause` is none or not given. This is part of a soccer player's contract dealing with being bought out by another team. After you have dropped them, see how many rows are remaining.**


```python
# Code here to drop rows with missing values for 'Release Clause'

```


```python
# Code here to check how many rows are left 

```

**3) Convert the `Release Clause` Price from Euros to Dollars**

Now that there are no missing values, we can change the values in the `Release Clause` column from Euro to Dollar amounts.

Assume the current exchange rate is
`1 Euro = 1.2 Dollars`


```python
# Code here to convert the column of euros to dollars
```


```python

```
