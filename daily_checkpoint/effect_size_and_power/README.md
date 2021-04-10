# Power and Effect Size Checkpoint


```python
#run this cell without changes

import numpy as np
import pandas as pd
from statsmodels.stats import power
from scipy import stats
```

#### 1. What is the relationship between power and the false-negative rate, $\beta$ ?


```python

```

#### 2. Calculate Cohen's *d* for the following data.  

Suppose we have the following samples of ant body lengths in centimeters, taken from two colonies that are distinct but spatially close to one another.

Calculate Cohen's *d* for these groups.

(The variance of each group has been provided for you.)


```python
#run this cell without changes

group1 = np.array([2.101, 2.302, 2.403])
group2 = np.array([2.604, 2.505, 2.506])

group1_var = np.var(group1, ddof=1)
group2_var = np.var(group2, ddof=1)
```

#### 3. Is this a large effect size? What conclusion should we draw about these two samples?


```python

```

#### 4. We decide we want to collect more data to have a more robust experiment. 

#### Given the current effect size, calculate how many observations we need (of each ant colony) if we want our test to have a power rating of 0.9 and a false-positive rate of 5%.


```python

```

Suppose we gather more data on our ants and then re-visit our calculations. 


```python
#run this cell without changes

col1 = pd.read_csv('data/colony1')
col2 = pd.read_csv('data/colony2')
```

#### 5. Do a two-tailed t-test on the lengths of ants in these two colonies.


```python

```

#### 6. What should we conclude about these two collections of ant body lengths?


```python

```
