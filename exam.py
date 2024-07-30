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

pass

# ### Exercise 2 (max 3 points)
#
# Plot the histograms (the number of bins should be 20 in both cases) of the `NR1_N` and `NR2A_N` values on the same figure.
# Add a legend to distinguish the plots.

pass

# ### Exercise 3 (max 3 points)
#
# Make a figure with two columns of plots. In the first column plot together (contrast) the histograms of `NR2A_N` for the two genotypes in the dataset. In the second column plot together (contrast) the histograms (20 bins) of `NR2A_N` for the two treatments in the dataset. Use density histograms to make the diagrams easy to compare; add proper titles and legends.
#

pass

# ### Exercise 4 (max 5 points)
#
# Make a (huge) figure with the histograms plotted in the previous exercise for all the proteins whose name starts with 'p' (there are 22 such columns), each in a different row of the figure (the figure will have 2 columns and 22 rows; to make it readable set the `figsize` to `(5, 3*22)`). 
#

pass

# ### Exercise 5 (max 7 points)
#
# Define a function `evodd_digits` that takes a float number and an even/odd flag: when the flag is True, the function should return all the even digits of the decimal representation of the number, otherwise it returns the odd ones. For example, if the number is 1.2345 and the flag is True, the function should return 24.  Think carefully about which return type is the most appropriate. 
#
# To get the full marks, you should declare correctly type hints and add a test within a doctest string.

pass

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

pass

# ### Exercise 7 (max 4 points)
#
# Draw the scatterplot of the standardized values of `'Bcatenin_N'` vs. (also standardized) `'Tau_N'` for the class 't-CS-s'. The standard value $z$ corresponding to a value $v$ taken from a series with mean $\bar v$ and standard deviation $\sigma$ is: $z = \frac{v - \bar v}{\sigma}$.

pass

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

pass