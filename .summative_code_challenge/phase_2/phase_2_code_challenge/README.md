# Mod 2 Code Challenge: Northwind Database

This assessment is designed to test your understanding of these areas:

1. Data Engineering
    - Interpreting an entity relationship diagram (ERD) of a real-world SQL database
    - Importing data from a real-world SQL database
    - Data manipulation in SQL and/or Pandas
2. Inferential Statistics
    - Set up a hypothesis test
    - Execute a hypothesis test
    - Interpret the results of a hypothesis test in order to answer a business question

Create a new Jupyter notebook to complete the challenge and show your work. Make sure that your code is clean and readable, and that each step of your process is documented. For this challenge each step builds upon the step before it. If you are having issues finishing one of the steps completely, move on to the next step to attempt every section.

### Setting up a Hypothesis Test

The business question you are trying to answer is:

> Is the mean `Quantity` of a product ordered *greater* when the product is discounted, compared to when it is not?

In the Jupyter Notebook, document:

 - The null hypothesis H<sub>0</sub> and alternative hypothesis H<sub>A</sub>
 - What does it mean to make `Type I` and `Type II` errors in this context?

### Importing Data

Contained in this repo is a SQLite database named ***northwind_db.sqlite***.  This database represents a subset of of the Northwind database — a free, open-source dataset created by Microsoft.  It contains the sales data for a fictional company called Northwind Traders.  The full ERD is below.

For your main task, you are only required to utilize the **OrderDetail** table.

Using SQL and/or Pandas, import the data contained in the OrderDetail table to begin data cleaning and analysis.

![Northwind ERD](northwind_erd.png)

### Preprocessing Data

Before executing a statistical test, some preprocessing is needed to set up the framing of *when the product is discounted*.  Specifically, create two array-like variables `discounted` and `not_discounted`, distinguished from each other based on the values in the `Discount` column and containing values from the `Quantity` column.  (In other words, two "lists" of quantities, although those "lists" can be Pandas Series objects, NumPy arrays, base Python lists, or any other array-like.)  The definition of "discounted" is that `Discount` is greater than 0.

### Executing a Hypothesis Test

Run a statistical test on the two samples, `discounted` and `not_discounted`.  Use a significance level of &alpha; = 0.05.

***Note:*** treat the entire collections `discounted` and `not_discounted` as "samples" of their respective populations.  You do not need to take further samples from those collections.

Write a few sentences explaining why you have chosen to run the statistical test you have chosen to run.

You may import the functions stored in the `flatiron_stats.py` file to help perform your hypothesis tests. It contains the stats functions from the Learn.co lessons: `welch_t(a,b)`, `welch_df(a, b)`, and `p_value_welch_ttest(a, b, two_sided=False)`.

Note that `scipy.stats.ttest_ind(a, b, equal_var=False)` performs a two-sided Welch's t-test and that p-values derived from two-sided tests are two times the p-values derived from one-sided tests. See the [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) for more information.

### Interpreting Results

What do the results of your hypothesis test indicate?  Can you reject the null hypothesis, and what does this mean from a business context?  Consider utilizing visualizations to support your recommendation to Northwind Trading.

# Deliverables Checklist

(Using markdown, mark items as complete in this checklist by changing `[ ]` to `[x]`)

Your main deliverable is a Jupyter notebook containing the following clearly labeled:

 - [ ] Documentation of the null and alternative hypotheses
 - [ ] Documentation of what Type I and Type II errors mean in this context
 - [ ] An array-like variable `discounted` that contains the `Quantity` values for records with `Discount` values greater than 0
 - [ ] An array-like variable `not_discounted` that contains the `Quantity` values for records with `Discount` values equal to 0
 - [ ] A hypothesis test that answers the business question: *Is the mean `Quantity` of a product ordered greater when the product is discounted, compared to when it is not?*
 - [ ] A short paragraph detailing your findings

## Bonus

NOTE: Please do not attempt this section until you have fully completed the main sections. `git add` and `git commit` your code from the previous sections before continuing.

Your previous analysis indicates whether customers are buying higher quantities per order for discounted products.  But how does that impact the **profit margin per order**?

For the purpose of this analysis, profit margin per order is calculated as:

(`sale price` * `quantity per unit`) - `wholesale price`

 -  `sale price` is defined as the discounted `OrderDetail` `UnitPrice`, i.e.
    - `UnitPrice` * (1 - `Discount`)
 - `quantity per unit` is defined as the `QuantityPerUnit` from `Product`
 - `wholesale price` is defined as the `Product` `UnitPrice`

The question to answer is:

> Is the mean profit margin per order greater when the product is discounted, compared to when it is not?

Answer this question descriptively first (comparing the two means), then set up and execute a hypothesis test if you have time.
