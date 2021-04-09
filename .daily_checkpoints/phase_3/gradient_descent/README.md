# Calculus, Cost Function, and Gradient Descent

![best fit line](visuals/best_fit_line.png)

The best fit line that goes through the scatterplot up above can be generalized in the following equation: $$y = mx + b$$



Of all the possible lines, we can prove why that particular line was chosen using the plot down below:

![](visuals/cost_curve.png)

where RSS is defined as the residual sum of squares:

$$ 
\begin{align}
RSS &= \sum_{i=1}^n(actual - expected)^2 \\
&= \sum_{i=1}^n(y_i - \hat{y})^2 \\
&= \sum_{i=1}^n(y_i - (mx_i + b))^2
\end{align}
$$ 

### 1. What is a more generalized name for the RSS curve above? How is it related to machine learning models?


```python
# Your answer here
```

### 2. Would you rather choose a $m$ value of 0.08 or 0.05 from the RSS curve up above?   What is the relation between the position on the cost curve, the error, and the slope `m` of the regression?


```python
# Your answer here
```

![](visuals/gd.png)

### 3. Using the gradient descent visual from above, explain why the distance between each step is getting smaller as more steps occur with gradient descent.


```python
# Your answer here
```

### 4. What is the purpose of a learning rate in gradient descent? Explain how a very small and a very large learning rate would affect the gradient descent.


```python
# Your answer here
```


```python

```
