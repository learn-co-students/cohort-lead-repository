### 1. Describe the difference between content-based and collaborative-filtering recommendation systems. Be sure to explain the role that *similarity* plays in each.


```python
# Your answer here


```

### 2. What is the singular value decomposition and how is it relevant to recommendation systems?


```python
# Your answer here


```

### 3. How is the method of Alternating Least Squares related to the method of Ordinary Least Squares (that we use in linear regression)?


```python
# Your answer here


```

### 4. Calculate Cosine Similarity

Suppose we have a database that represents films as rows, the columns representing the presence or absence of certain actors, directors, genres, etc. Below are the first three rows in this database:


```python
import numpy as np

films = np.array([[0, 1, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1]]).reshape(3, -1)
films
```




    array([[0, 1, 0, 1, 1, 1],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1, 1]])



Cristian was a big fan of the movie represented by Row \#1 but hasn't seen either of the other two. Based on cosine similarity, should we recommend the movie represented by Row \#2 or the movie represented by Row \#3?

Recall that the cosine of the angle $\theta$ between two vectors $\vec{v_1}$ and $\vec{v_2}$ can be expresed in terms of the dot product of the vectors and the vectors' magnitudes:

$\cos(\theta) = \frac{\vec{v_1}\cdot\vec{v_2}}{|\vec{v_1}||\vec{v_2}|}$


```python
# Your answer here


```


```python

```
