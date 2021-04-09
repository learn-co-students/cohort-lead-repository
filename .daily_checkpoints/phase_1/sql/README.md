## SQL and Relational Databases

In this checkpoint you will be exploring a Pokemon dataset that has been put into SQL tables. Pokemon are fictional creatures from the [Nintendo franchise](https://en.wikipedia.org/wiki/Pok%C3%A9mon) of the same name.

Some Pokemon facts that might be useful:
* The word "pokemon" is both singular and plural. You may refer to "one pokemon" or "many pokemon".
* Pokemon have attributes such as a name, weight, and height.
* Pokemon have one or multiple "types". A type is something like "electric", "water", "ghost", or "normal" that indicates the abilities that pokemon may possess.
* The humans who collect pokemon are called "trainers".

The schema of `pokemon.db` is as follows:

<img src="data/pokemon_db.png" alt="db schema" style="width:500px;"/>

Assign your SQL queries as strings to the variables `q1`, `q2`, etc. and run the cells at the end of this section to print your results as Pandas DataFrames.

- `q1`: Find all the pokemon on the "pokemon" table. Display all columns.  

  
- `q2`: Find all the rows from the "pokemon_types" table where the type_id is 3.


- `q3`: Find all the rows from the "pokemon_types" table where the associated type is "water". Do so without hard-coding the id of the "water" type, using only the name.


- `q4`: Find the names of all pokemon that have the "psychic" type.


- `q5`: Find the average weight for each type. Order the results from highest weight to lowest weight. Display the type name next to the average weight.


- `q6`: Find the names and ids of all the pokemon that have more than 1 type.


- `q7`: Find the id of the type that has the most pokemon. Display type_id next to the number of pokemon having that type. 


**Important note on syntax**: use `double quotes ""` when quoting strings **within** your query and wrap the entire query in `single quotes ''`.

**DO NOT MODIFY THE PYTHON CODE BELOW (e.g. `pd.read_sql`). YOU ONLY NEED TO MODIFY THE SQL QUERY STRINGS.**


```python
# Run this cell without changes
import pandas as pd
import sqlite3
```


```python
# Run this cell without changes
cnx = sqlite3.connect('data/pokemon.db')
```

### Question 1: Find all the pokemon on the "pokemon" table. Display all columns.


```python
# Enter appropriate SQL code within the quotes
q1 = ''
pd.read_sql(q1, cnx)
```

### Question 2: Find all the rows from the "pokemon_types" table where the type_id is 3.


```python
# Enter appropriate SQL code within the quotes
q2 = ''
pd.read_sql(q2, cnx)
```

### Question 3: Find all the rows from the "pokemon_types" table where the associated type is "water". Do so without hard-coding the id of the "water" type, using only the name.


```python
# Enter appropriate SQL code within the quotes
q3 = '''

'''
pd.read_sql(q3, cnx)
```

### Question 4: Find the names of all pokemon that have the "psychic" type.


```python
# Enter appropriate SQL code within the quotes
q4 = '''

'''
pd.read_sql(q4, cnx)
```

### Question 5: Find the average weight for each type. Order the results from highest weight to lowest weight. Display the type name next to the average weight.


```python
# Enter appropriate SQL code within the quotes
q5 = '''

'''
pd.read_sql(q5, cnx)
```

### Question 6: Find the names and ids of all the pokemon that have more than 1 type. 


```python
# Enter appropriate SQL code within the quotes
q6 = '''

'''
pd.read_sql(q6, cnx)
```

### Question 7: Find the id of the type that has the most pokemon. Display type_id next to the number of pokemon having that type. 


```python
# Enter appropriate SQL code within the quotes
q7 = '''

'''
pd.read_sql(q7, cnx)
```
