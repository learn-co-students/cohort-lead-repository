# APIs and Point Estimates


```python
import pandas as pd
from matplotlib import pyplot as plt
import requests
```

## APIs for Data Collection

For this exercise, you're going to collect all [wage data from the City of Seattle](https://data.seattle.gov/City-Business/City-of-Seattle-Wage-Data/2khk-5ukd).  There are approximately 12,000 employees in this dataset, although the exact numbers can be expected to shift.

This example code shows how to retrieve the default number of records (1000):


```python
# Run this cell without changes
BASE_URL = "https://data.seattle.gov/resource/2khk-5ukd.json"
base_response = requests.get(BASE_URL)
base_response_json = base_response.json()
len(base_response_json)
```

### 1) In the following cell, create a URL where the `$limit` URL parameter has been modified to be 15000 rather than the default

See the [Socrata API docs](https://dev.socrata.com/docs/queries/) for more information about the available parameters


```python
# Replace None with appropriate code to use in your answer
URL_WITH_LIMIT = None
full_response = requests.get(URL_WITH_LIMIT)
full_response_json = full_response.json()

# Should print a number around 12k
len(full_response_json)
```

### 2) Create a Pandas DataFrame with the data from `full_response_json`


```python
# Replace None with appropriate code to use in your answer
df = None
```


```python
# Run this cell without changes
df.head()
```


```python
# Run this cell without changes
df.info()
```

### 3) As you can see, `hourly_rate` is currently type `object`.  Convert this column into a numeric value


```python
# Your code here
```


```python
# Run this cell without changes
df.info()
```

## Point Estimates

Now that we have our data, let's compare statistics from a sample of the population to the overall population

### 4) Select a random sample of 500 city employees
Use a `random_state` of 2020 for reproducibility


```python
# Replace None with appropriate code to use in your answer
df_sample = None
```


```python
# Run this cell without changes
df_sample.head()
```

The following code will plot a histogram of the `hourly_rate` feature of the full DataFrame as well as `df_sample`


```python
# Run this cell without changes
plt.style.use('seaborn-darkgrid')
fig, axes = plt.subplots(ncols=2, figsize=(14, 6))

axes[0].hist(df_sample["hourly_rate"], bins=20)
axes[0].set_title("Sample")
axes[0].set_xlabel("Hourly Wage")
axes[0].set_ylabel("Count of Records in Sample")
axes[0].set_xlim(0, 175)

axes[1].hist(df["hourly_rate"], bins=20)
axes[1].set_title("Population")
axes[1].set_xlabel("Hourly Wage")
axes[1].set_ylabel("Count of Records in Full Population")
axes[0].set_xlim(0, 175)

fig.suptitle("Distribution of Hourly Wages in the City of Seattle", fontsize=18)
plt.show()
```

### 5) Compare statistics for the sample compared to the overall population

Discuss how `hourly_rate` in the sample compares to `hourly_rate` in the population:
- Compare measures of central tendency and spread for each

- Make an argument that `hourly_rate` in the sample of 500 employees either does or does not adequately represent `hourly_rate` in the population.  

You can include insights from the graphs above in your analysis


```python
# Your code here
```


```python
"""
Your written answer here
"""
```


```python

```
