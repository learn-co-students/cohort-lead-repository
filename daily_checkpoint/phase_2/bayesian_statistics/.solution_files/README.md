# Bayesian Statistics

### Puppy Questions

Thomas wants to get a new puppy 🐕 🐶 🐩 


<img src="https://media.giphy.com/media/rD8R00QOKwfxC/giphy.gif" />

He can choose to get his new puppy either from the pet store or the pound. The probability of him going to the pet store is $0.2$. 

He can choose to get either a big, medium or small puppy.

If he goes to the pet store, the probability of him getting a small puppy is $0.6$. The probability of him getting a medium puppy is $0.3$, and the probability of him getting a large puppy is $0.1$.

If he goes to the pound, the probability of him getting a small puppy is $0.1$. The probability of him getting a medium puppy is $0.35$, and the probability of him getting a large puppy is $0.55$.


#### 1) What is the probability of Thomas getting a small puppy?


```python
# P(Small) = P(Small|Pet Store) + P(Small|Pound) = 0.2*0.6 + 0.8*0.1 = 0.2
ans1 = 0.2

ans1
```




    0.2



#### 2) Given that he got a large puppy, what is the probability that Thomas went to the pet store?


```python
"""
P(Pet Store|Large)  = P(Large|Pet Store)*P(Pet Store) / P(Large) 
                    = 0.1*0.2 / (0.1*0.2 + 0.55*0.8)
                    = 0.02 / 0.46 = 0.04348
"""
ans2 = 0.02/0.46
ans2
```




    0.043478260869565216



### Med Question

A medical test is designed to diagnose a certain disease. The test has a false positive rate of 10%, meaning that 10% of people without the disease will get a positive test result. The test has a false negative rate of 2%, meaning that 2% of people with the disease will get a negative result. Only 1% of the population has this disease.


#### 3) If a patient receives a positive test result, what is the probability that they actually have the disease? Show how you arrive at your answer.


```python

P_1 = 0.98*0.01/(0.98*0.01 + 0.10*0.99)
P_1

"""
Dis: Has the disease
Pos: Has a positive test result

P(Dis|Pos) = P(Pos|Dis)*P(Dis)/P(Pos) 
           = P(Pos|Dis)*P(Dis)/[P(Pos|Dis)*P(Dis) + P(Pos|No Dis)*P(No Dis)]
           = 0.98*0.01/(0.98*0.01 + 0.10*0.99)
           = 0.090 = 9%
"""
```




    0.09007352941176469


