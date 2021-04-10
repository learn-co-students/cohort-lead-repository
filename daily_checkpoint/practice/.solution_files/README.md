# Data Science Practice Daily Checkpoint

This assignment is designed to test your understanding of the daily checkpoint process. It covers:

 - Anwering a code question
 - Answering a short answer question
 
Read the instructions carefully. You will be asked both to write code and to respond to a short answer question.

**Note on daily checkpoint short answer questions:** Please use your own words. The expectation is that you have not copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, you should do your best to communicate yourself clearly.

## Imports

You will usually have to import different packages into a checkpoint or other material we give you

Don't worry if you don't have all the mechanics about what that means at this point

Just run the cell below, which allows the test code to function

## Question 1

Adapt the `greet_user` function to do the following:

 - take a name as an argument
 - print "Greetings! I am `name` and I am pleased to make your acquaintance!"

Then in the next cell, assign the name "Gene Hackman" to the variable `name`.

Run all cells to confirm the result.


```python

def greet_user(name):
    """
    this function takes a string name as an argument 
    and prints a greeting including that name 
    """
    
    print(f"Greetings! I am {name} and I am pleased to make your acquaintance!")

name = "Gene Hackman"

greet_user(name)
```

## Question 2

Please explain the purpose of daily checkpoints and code challenges in the Flatiron School Data Science program.


```python
"""The purpose of assessments are to assess a student's comprehension and application of our Data Science program's key content.
Daily checkpoints are formative, in-depth assessments
Code challenges are summative, end-to-end assessments"""
```
