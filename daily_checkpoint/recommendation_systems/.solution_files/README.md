### 1. Describe the difference between content-based and collaborative-filtering recommendation systems. Be sure to explain the role that *similarity* plays in each.


```python

# In content-based recommendation systems, items are recommended to a user based on their similarity
# to items already consumed and enjoyed. So this depends on a notion of similarity that applies to
# the items themselves.

# In collaborative systems, items are recommended to a user based on how similar users rated them. So
# this depends on a notion of similarity that applies to the users.
```

### 2. What is the singular value decomposition and how is it relevant to recommendation systems?


```python

# The singular value decomposition is a decomposition of a matrix related to eigendecomposition.
# But it is more general since it applies to non-square matrices. Because the SVD can be used to
# solve a least-squares problem, it is relevant to recommendation system theory. SVD can also be
# used for dimensionality reduction in much the same way that PCA can. PCA invites us to choose
# eigenvectors corresponding to large eigenvalues. We might just as well choose singular vectors
# corresponinding to large singular values.
```

### 3. How is the method of Alternating Least Squares related to the method of Ordinary Least Squares (that we use in linear regression)?


```python

# Alternating Least Squares is a method to fill in a ratings matrix where the rows represent users
# (doing the rating) and the columns represent items (being rated). The idea is to factor the
# matrix into a users matrix and an items matrix. And in order to generate these matrices, we
# take turns holding one constant and solving the least-squares problem for the other.
```

### 4. Calculate Cosine Similarity

Suppose we have a database that represents films as rows, the columns representing the presence or absence of certain actors, directors, genres, etc. Below are the first three rows in this database:

Cristian was a big fan of the movie represented by Row \#1 but hasn't seen either of the other two. Based on cosine similarity, should we recommend the movie represented by Row \#2 or the movie represented by Row \#3?

Recall that the cosine of the angle $\theta$ between two vectors $\vec{v_1}$ and $\vec{v_2}$ can be expresed in terms of the dot product of the vectors and the vectors' magnitudes:

$\cos(\theta) = \frac{\vec{v_1}\cdot\vec{v_2}}{|\vec{v_1}||\vec{v_2}|}$


```python

mag1 = np.sqrt(4)
mag2 = np.sqrt(3)
mag3 = np.sqrt(2)

print('Similarity between Film #1 and Film #2:' + str(films[0].dot(films[1]) / (mag1 * mag2)))
print('Similarity between Film #1 and Film #3:' + str(films[0].dot(films[2]) / (mag1 * mag3)))

# Since Film #3 is more similar to Film #1 than Film #2 is, we should recommend Film #3.
```

    Similarity between Film #1 and Film #2:0.5773502691896258
    Similarity between Film #1 and Film #3:0.7071067811865475

