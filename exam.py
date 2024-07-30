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

data = pd.read_csv('mice.csv', index_col='MouseID')

data.head()

# ### Exercise 2 (max 3 points)
#
# Plot the histograms (the number of bins should be 20 in both cases) of the `NR1_N` and `NR2A_N` values on the same figure.
# Add a legend to distinguish the plots.

# +
fig, ax = plt.subplots()

for p in ('NR1_N', 'NR2A_N'):
    data[p].hist(ax=ax, bins=20, label=p, alpha=.7)
_ = fig.legend()
# -

# ### Exercise 3 (max 3 points)
#
# Make a figure with two columns of plots. In the first column plot together (contrast) the histograms of `NR2A_N` for the two genotypes in the dataset. In the second column plot together (contrast) the histograms (20 bins) of `NR2A_N` for the two treatments in the dataset. Use density histograms to make the diagrams easy to compare; add proper titles and legends.
#

# +
fig, ax = plt.subplots(ncols=2)


col = 'NR2A_N'
for i, f in enumerate(['Genotype', 'Treatment']):
    for g in data[f].unique():
        ax[i].hist(data[data[f] == g][col], density=True, bins=20, label=g, alpha=.7)
        ax[i].legend()
        ax[i].set_title(f)
fig.tight_layout()
# -

# ### Exercise 4 (max 5 points)
#
# Make a (huge) figure with the histograms plotted in the previous exercise for all the proteins whose name starts with 'p' (there are 22 such columns), each in a different row of the figure (the figure will have 2 columns and 22 rows; to make it readable set the `figsize` to `(5, 3*22)`).
#

# +
fig, ax = plt.subplots(ncols=2, nrows=22, figsize=(5, 3*22))


for j, col in enumerate([c for c in data.columns if c.startswith('p')]):
    for i, f in enumerate(['Genotype', 'Treatment']):
        for g in data[f].unique():
            ax[j, i].hist(data[data[f] == g][col], density=True, bins=20, label=g, alpha=0.7)
            ax[j, i].legend()
            ax[j, i].set_title(f'{col} ({f})')
fig.tight_layout()


# -

# ### Exercise 5 (max 7 points)
#
# Define a function `evodd_digits` that takes a float number and an even/odd flag: when the flag is True, the function should return all the even digits of the decimal representation of the number, otherwise it returns the odd ones. For example, if the number is 1.2345 and the flag is True, the function should return 24.  Think carefully about which return type is the most appropriate.
#
# To get the full marks, you should declare correctly type hints and add a test within a doctest string.

def evodd_digits(x: float, even: bool = True) -> str:
    """Return all the even (or odd if even is False) digits of the decimal representation of the number x.

    >>> evodd_digits(1.2345)
    '24'

    >>> evodd_digits(1.2345, even = False)
    '135'

    >>> evodd_digits(0.2345)
    '024'

    """
    ris = ''
    for c in (d for d in str(x) if d.isnumeric()):
        if even and int(c) % 2 == 0:
            ris += c
        elif not even and int(c) % 2 != 0:
            ris += c
    return ris


# +
# You can test your docstrings by uncommenting the following two lines

import doctest
doctest.testmod()
# -

# ### Exercise 6 (max 5 points)
#
# Add a column `NR_evodd` with the results of the function defined in Exercise 5 computed, for each row, on the values of the columns `NR1_N`, `NR2A_N` concatenated in this order with the flag on True.
#
# To get full marks, avoid the use of explicit loops.
#

data['NR_evodd'] = data['NR1_N'].map(str).str.cat(data['NR2A_N'].map(str)).map(evodd_digits)


# ### Exercise 7 (max 4 points)
#
# Draw the scatterplot of the standardized values of `'Bcatenin_N'` vs. (also standardized) `'Tau_N'` for the class 't-CS-s'. The standard value $z$ corresponding to a value $v$ taken from a series with mean $\bar v$ and standard deviation $\sigma$ is: $z = \frac{v - \bar v}{\sigma}$.

# +
def standardize(x: pd.Series) -> pd.Series:
    """For each value assess its distance from the mean, w.r.t. the standard deviation.

    >>> standardize(pd.Series([-1, 0, 1])).tolist() # mean: 0, devstd: 1
    [-1.0, 0.0, 1.0]
    """
    return (x - x.mean()) / x.std()

fig, ax = plt.subplots()
d = data[data['class'] == 't-CS-s']

xd = standardize(d['Bcatenin_N'])
yd = standardize(d['Tau_N'])

ax.scatter(xd, yd)
ax.set_title('Tau_N vs. Bcatenin_N for t-CS-s')
fig.tight_layout()
# -

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
    a = pm.Normal('alpha', mu=0, sigma=0.2)
    b = pm.Normal('beta', mu=0, sigma=0.5)
    g = pm.Exponential('gamma', 1)

    pm.Normal("t", mu=a+b*xd, sigma=g, observed=yd)

    idata = pm.sample()
az.summary(idata)
