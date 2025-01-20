# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Programming in Python
# ## Exam: July 30, 2024
#
# You can solve the exercises below by using standard Python 3.11 libraries, NumPy, Matplotlib, Pandas, PyMC.
# You can browse the documentation: [Python](https://docs.python.org/3.11/), [NumPy](https://numpy.org/doc/1.26/index.html), [Matplotlib](https://matplotlib.org/3.8.2/users/index.html), [Pandas](https://pandas.pydata.org/pandas-docs/version/2.1/index.html), [PyMC](https://www.pymc.io/projects/docs/en/v5.10.3/api.html).
# You can also look at the slides or your code on [GitHub](https://github.com). 
#
# **It is forbidden to communicate with others or "ask questions" online (i.e., stackoverflow is ok if the answer is already there, but you cannot ask a new question or use ChatGPT or similar products)**
#
# To test examples in docstrings use
#
# ```python
# import doctest
# doctest.testmod()
# ```
#

import numpy as np
import pandas as pd             # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pymc as pm               # type: ignore
import arviz as az              # type: ignore

# ### Exercise 1 (max 2 points)
#
# The file [mice.csv](./mice.csv) (source: https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) contains the expression levels of 77 proteins/protein modifications that produced detectable signals in the nuclear fraction of cortex. 
# The eight classes of mice are described based on features such as genotype, behavior and treatment. According to genotype, mice can be control or trisomic. According to behavior, some mice have been stimulated to learn (context-shock) and others have not (shock-context) and in order to assess the effect of the drug memantine in recovering the ability to learn in trisomic mice, some mice have been injected with the drug and others have not.
#
# Load the data in a pandas dataframe using the `MouseID` column as the index.

df = pd.read_csv('mice.csv')

#inplace=True: This modifies the original DataFrame in place. If you omit inplace=True,
#it returns a new DataFrame with the index set, and you would need to assign it back to
#df like this: df = df.set_index('column_name').

df.set_index('MouseID', inplace=True)

# ### Exercise 2 (max 3 points)
#
# Plot the histograms (the number of bins should be 20 in both cases) of the `NR1_N` and `NR2A_N` values on the same figure.
# Add a legend to distinguish the plots.

fig, ax = plt.subplots(nrows=1, ncols=2)
df.hist('NR1_N',bins= 20,alpha=0.5, label='NR1_N', ax=ax[0])
df.hist('NR2A_N',bins= 20,alpha=0.5, label='NR2A_N', ax=ax[1])


ax[0].set_ylabel('Expression level')
ax[0].set_ylabel('Expression level')
plt.legend()

plt.show()
# ### Exercise 3 (max 3 points)
#
# Make a figure with two columns of plots. In the first column plot together (contrast) the histograms of `NR2A_N` for the two genotypes in the dataset. In the second column plot together (contrast) the histograms (20 bins) of `NR2A_N` for the two treatments in the dataset. Use density histograms to make the diagrams easy to compare; add proper titles and legends.
#

fig, ax = plt.subplots(nrows=1, ncols=2)
for gen, group in df.groupby('Genotype'):
    group['NR2A_N'].plot(kind='hist',bins=20, density=True, alpha=0.5, label=gen, ax=ax[0])
for t, group in df.groupby('Treatment'):
    group['NR2A_N'].plot(kind='hist', bins=20, density=True, alpha=0.5, label=t, ax=ax[1])
ax[0].set_ylabel('Expression level')
ax[1].set_ylabel('Expression level')
ax[0].legend(title='Genotype')
ax[1].legend(title='Treatment')
plt.show()

# ### Exercise 4 (max 5 points)
#
# Make a (huge) figure with the histograms plotted in the previous exercise for all the proteins whose name starts with 'p' (there are 22 such columns), each in a different row of the figure (the figure will have 2 columns and 22 rows; to make it readable set the `figsize` to `(5, 3*22)`). 
#

fig, ax = plt.subplots(nrows=22, ncols=2, figsize=(2, 3*300))

col_list=df.columns.tolist()
p_list = []
for i in col_list:
    if list(i)[0]=='p':
        p_list.append(i)
# Plot histograms for each protein
for i, col in enumerate(p_list):
    # Plot histograms for NR2A_N by Genotype
    for genotype, group in df.groupby('Genotype'):
        group[col].plot(kind='hist', bins=20, density=True, alpha=0.5, ax=ax[i, 0])

    
# Plot histograms for NR2A_N by Treatment
    for treatment, group in df.groupby('Treatment'):
        group[col].plot(kind='hist', bins=20, density=True, alpha=0.5, ax=ax[i, 1])

# Show the plot
plt.show()
    
    


# ### Exercise 5 (max 7 points)
#
# Define a function `evodd_digits` that takes a float number and an even/odd flag: when the flag is True, the function should return all the even digits of the decimal representation of the number, otherwise it returns the odd ones. For example, if the number is 1.2345 and the flag is True, the function should return 24.  Think carefully about which return type is the most appropriate. 
#
# To get the full marks, you should declare correctly type hints and add a test within a doctest string.

def evodd_digits(number: float, even_flag: bool)->str:
    # Convert the number to a string and remove the decimal point
    number_str = str(number).replace(".", "")
    
    # Convert the string to a list of digits
    digits = list(number_str)
    
    # Filter the digits based on the even_flag
    if even_flag:
        result = [d for d in digits if d.isdigit() and int(d) % 2 == 0]
    else:
        result = [d for d in digits if d.isdigit() and int(d) % 2 != 0]
    
    # Join the filtered digits and return as a string
    return ''.join(result)



# +
# You can test your docstrings by uncommenting the following two lines

# import doctest
# doctest.testmod()
# -

# ### Exercise 6 (max 5 points)
#
# Add a column `NR_evodd` with the results of the function defined in Exercise 5 computed, for each row, on the values of the columns `NR1_N`, `NR2A_N` concatenated in this order with the flag on True.
#
# To get full marks, avoid the use of explicit loops.
#

# Fixing the code with minimal changes
df['NR_evodd1'] = df.apply(lambda row: evodd_digits(float(str(row['NR1_N']).replace('.', '')), True), axis=1)
df['NR_evodd2'] = df.apply(lambda row: evodd_digits(float(str(row['NR2A_N']).replace('.', '')), True), axis=1)
df['NR_evodd'] = df['NR_evodd1'] + ',' + df['NR_evodd2']

# ### Exercise 7 (max 4 points)
#
# Draw the scatterplot of the standardized values of `'Bcatenin_N'` vs. (also standardized) `'Tau_N'` for the class 't-CS-s'. The standard value $z$ corresponding to a value $v$ taken from a series with mean $\bar v$ and standard deviation $\sigma$ is: $z = \frac{v - \bar v}{\sigma}$.

def standardize_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    mean_value = df[column_name].mean()
    std_value = df[column_name].std()
    standardized_column = (df[column_name] - mean_value) / std_value
    return standardized_column

df['standardized Bcatenin_N']=standardize_column(df, 'Bcatenin_N')
df['standardized Tau_N']=standardize_column(df, 'Tau_N')

for g, group in df.groupby('class'):
    if g == 't-CS-s':
        plt.scatter(df['standardized Bcatenin_N'],df['standardized Tau_N'], label='Bcatenin_N vs Tau_N')
        plt.xlabel('standardized Bcatenin_N')
        plt.ylabel('standardized Tau_N')
        plt.legend()       

plt.show()

# ### Exercise 8 (max 4 points)
#
# Consider this statistical model:
#
# - a parameter $\alpha$ is normally distributed with $\mu = 0$ and $\sigma = 0.2$ 
# - a parameter $\beta$ is normally distributed with $\mu = 0$ and $\sigma = 0.5$ 
# - a parameter $\gamma$ is exponentially distributed with $\lambda = 1$
# - the observed standardized `Tau_N` for class 't-CS-s' is normally distributed with standard deviation $\gamma$ and a mean given by $\alpha + \beta \cdot M$ (where $M$ is the correspondig value of standardized `Bcatenin_N` for class 't-CS-s').
#
# Code this model with pymc, sample the model, and print the summary of the resulting estimation by using `az.summary`. 

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sigma=0.2)
    beta = pm.Normal("beta", mu=0, sigma=0.5)
    gamma = pm.Exponential("gamma", lam=1)
    
    mu = alpha + beta * df['standardized Bcatenin_N']
    observed_Tau = pm.Normal("observed Tau", mu=mu, sigma=gamma, observed=raw['standardized Tau_N'])
    
    trace = pm.sample(1000, tune=1000, return_inferencedata=True)
    
summary = az.summary(trace)
print(summary)