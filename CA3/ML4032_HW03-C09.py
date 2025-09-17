#!/usr/bin/env python
# coding: utf-8

# # **Machine Learning Homework: Regression Analysis and Nonlinear Modeling**
# 
# ---
# 
# ## **Objective**  
# Students will explore linear and nonlinear regression techniques using the Ames Housing dataset. The assignment is divided into two parts: manual calculations and Python implementation. The goal is to understand regression fundamentals and practice modeling nonlinear relationships.  
# 

# ---
# ## **Part 1: Manual Regression Calculations**  
# ### **Step 1: Simple Linear Regression**  
# Given the following data points for `GrLivArea` (x) and `SalePrice` (y):  
# 
# | x (GrLivArea) | y (SalePrice) |
# |---------------|---------------|
# | 1500          | 200000        |
# | 1600          | 210000        |
# | 1700          | 220000        |
# | 1800          | 230000        |
# | 1900          | 240000        |
# 
# #### Tasks:
# - Compute the mean of x and y.
# - Calculate the slope (w) and intercept (b) using the least squares formula.
# - Write the regression equation: `y = wx + b`.
# - Predict the price for a house with `GrLivArea = 2000`.
# 

# ### **Step 2: Polynomial Regression (Manual Expansion)**  
# Using the same dataset, expand x to include a quadratic term: `x²`.  
# 
# | x | x² |
# |----|----|
# |1500|2250000|
# |1600|2560000|
# |1700|2890000|
# |1800|3240000|
# |1900|3610000|
# 
# #### Tasks:
# - Fit a second-degree polynomial regression manually using matrix notation.
# - Solve for coefficients using normal equations.
# - Compare predictions for `x = 2000` using linear vs. polynomial models.
# 

# ---
# ## **Part 2: Python Implementation**  
# ### **Step 1: Load and Explore the Ames Housing Dataset**  
# - Load the dataset from `https://raw.githubusercontent.com/rasbt/machine-learning-book/main/code/ch09/housing.csv`
# - Display basic statistics and correlation matrix.
# - Visualize `GrLivArea` vs `SalePrice` using scatter plots.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/rasbt/machine-learning-book/main/code/ch09/housing.csv")
df.plot(kind='scatter', x='GrLivArea', y='SalePrice')
plt.show()


# ### **Step 2: Linear Regression with Scikit-Learn**  
# - Fit a linear regression model using `GrLivArea` to predict `SalePrice`.
# - Print coefficients and intercept.
# - Evaluate model using MAE and R².
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X = df[['GrLivArea']].values
y = df['SalePrice'].values

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)
print("MAE:", mean_absolute_error(y, y_pred))
print("R²:", r2_score(y, y_pred))


# ### **Step 3: Polynomial Regression**  
# - Use `PolynomialFeatures` to add quadratic terms.
# - Fit and evaluate the polynomial model.
# - Compare performance with linear regression.
# 

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
y_poly_pred = lr_poly.predict(X_poly)

print("MAE (Poly):", mean_absolute_error(y, y_poly_pred))
print("R² (Poly):", r2_score(y, y_poly_pred))


# ### **Step 4: Tree-Based Regression**  
# - Fit a `DecisionTreeRegressor` and `RandomForestRegressor`.
# - Compare their performance with linear and polynomial models.
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

tree = DecisionTreeRegressor(max_depth=4)
tree.fit(X, y)
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, y)

print("R² (Tree):", r2_score(y, tree.predict(X)))
print("R² (RF):", r2_score(y, rf.predict(X)))


# ---
# ## **Submission Requirements**  
# 📌 Submit a `.ipynb` notebook with:
# - Manual calculations.
# - Python code and outputs.
# - A short reflection comparing model interpretability and performance.
# 
# ---
# *Ali Fahim*  
# *University of Tehran*  
# 
