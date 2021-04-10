
## Probability and Combinatorics



```python
# import the necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
```

### 1. Set Theory

Given the following probabilities:

$P(A) = 0.7$

$P(B) = 0.5$

$P(B|A) = 0.4$

Calculate the following probabilities and assign to the variables `ans1` and `ans2`, respectively, in the next cell:

Question 1: $P(A and B)$

Question 2: $P(A|B)$

Hint: draw a diagram!


```python
ans1 = 0.28
ans2 = 0.56

"""
Question 1:

We use the conditional probability formula: P(B|A) = P(A and B)/P(A)
P(A and B) = P(B|A)*P(A) = 0.4*0.7 = 0.28


Question 2:

P(A|B) = P(A and B)/P(B) = 0.28/0.5 = 0.56

"""
```

### 2. Card Combinatorics

A standard deck of playing cards consists of 52 cards in each of the four suits of spades, hearts, diamonds, and clubs. Each suit contains 13 cards: Ace, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, and King.
    
You have a standard deck of 52 cards and are asked the following questions:

Question 3: What is the probability of drawing a King or a Queen?

Question 4: How many possible 5-card combinations can be formed with this deck of 52 cards?

Answer the questions below:


```python
ans3 = 2/13
ans4 = 2598960

"""
Question 3:

P(King or Queen) = Number of Kings + Queens / Total Number of Cards = 8/52 = 2/13


Question 4:

Number of 5-card combinations = Number of ways to choose 5 from 52 = 52!/(5!*47!) = 2598960


"""
```
