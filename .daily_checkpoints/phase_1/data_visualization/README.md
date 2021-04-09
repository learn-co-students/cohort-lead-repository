## Data Visualization

In this checkpoint you will be using a FIFA dataset of soccer player statistics to create a visualization, then also interpreting a visualization we have created.

First, we import the relevant libraries


```python
# Run this cell without changes
import csv
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Next, we read data from a CSV and create `player_nationalities`, a list of dictionaries containing data about the number of FIFA soccer players from each country.  The countries are in random order.

This code also prints the first five dictionaries in `player_nationalities`, out of a total of 161


```python
# Run this cell without changes
player_nationalities = []
with open("data/player_nationalities.csv") as f:
    for row in csv.DictReader(f):
        row['count_of_players'] = int(row['count_of_players'])
        player_nationalities.append(dict(row))

for index in range(5):
    print(player_nationalities[index])
```

### 1a) Find the top 10 countries with the most players. 


```python
#Your answer here
```

### 1b) Create a bar chart showing the number of players in those 10 countries

Don't forget to add a **title** and **y axis label** to your charts.

A recommended figsize is (10, 6)

If you are unable to find the top 10 countries but want the chance to demonstrate your plotting skills use the first 10 randomly ordered countries: `top10 = player_nationalities[:10]`


```python
#Your answer here
```

**2) Below is code for a scatter plot for the player stats `Standing Tackle` and `Sliding Tackle`**


```python
# Run this cell without changes
player_stats = []
with open("data/player_stats.csv") as f:
    for row in csv.DictReader(f):
        row["standing_tackle"] = float(row["standing_tackle"])
        row["sliding_tackle"] = float(row["sliding_tackle"])
        player_stats.append(dict(row))
        
for index in range(5):
    print(player_stats[index])
```


```python
# Run this cell without changes
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_title('Standing Tackle vs. Sliding Tackle')
ax.set_xlabel('Standing Tackle')
ax.set_ylabel('Sliding Tackle')

x = [row["standing_tackle"] for row in player_stats]
y = [row["sliding_tackle"] for row in player_stats]

ax.scatter(x, y, alpha=0.2);
```

**How would you describe the relationship between these two features?**


```python
#Your answer here
```
