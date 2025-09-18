#!/usr/bin/env python
# coding: utf-8

# In the name of God

# # **Machine Learning Homework: Regression Analysis and Nonlinear Modeling**

# Seyed Mani Mirshabani 
# 810801080

# با توجه به اینکه قصد دارم در گیتهابم قرار بدم و بعدا به عنوان رزومه باشه با اجازه شما متن های نوتبوک رو به انگلیسی مینویسم (برای این کار از هوش مصنوعی استفاده نمیکنم و هرچیزی که متوجه شده باشم رو خودم به انگلیسی مینویسم برای همین پیشاپیش بابت اشتباهات مربوط به انگلیسی معذرت میخوام)

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

# ![1.1.1](1.jpg)
# 
# ![1.1.2](2.jpg)

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

# ![1.2.1](3.jpg)
# 
# ![1.2.2](3.jpg)

# ## **Part 2: Python Implementation**  

# ### **Step 1: Load and Explore the Ames Housing Dataset**  

# - Load the dataset from `https://raw.githubusercontent.com/rasbt/machine-learning-book/refs/heads/main/ch09/AmesHousing.txt`
# - Display basic statistics and correlation matrix.
# - Visualize `GrLivArea` vs `SalePrice` using scatter plots.

# For this section we will go a step beyond what is requested by checking the missing values and performing some basic EDA. The steps we will take will be as follows:
# 
# 1. Loading teh data.
# 2. Getting some quick info from the data.
# 3. Looking at correlations
# 4. Drawing the scatter plot
# 5. Looking at the distribution
# 6. Checking for outliers

# #### 2.1.1. Loading the data

# In this section we will load the data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 5)

url = "https://raw.githubusercontent.com/rasbt/machine-learning-book/refs/heads/main/ch09/AmesHousing.txt"

try:
    df = pd.read_csv(url, sep='\t', low_memory=False)
except Exception:
    df = pd.read_csv(url, sep=',', low_memory=False)

print("Loaded dataset shape:", df.shape)


# #### 2.1.2. Getting some quick information about the data

# In this section we get an overall idea by looking at the head, info, describe and missing values of the dataset.

# In[2]:


display(df.head(8))
print("\n--- Info ---")
df.info()

print("\n--- Numeric summary (describe) ---")
display(df.describe().T)

missing = df.isna().sum().sort_values(ascending=False)
print("\nTop missing-value columns (if any):")
display(missing[missing > 0].head(20))


# As we can see, we have quite a few missing values and we need to do something about them.
# 
# For handling these missing values we will drop the columns with too many missing values (400+) and impute the values on the rest (using the median for numerical values to avoid being effected by outliers and replacing the categorical values with 'Missing')

# In[3]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

threshold = 400

missing = df.isna().sum().sort_values(ascending=False)
print("Top missing counts:")
print(missing.head(30))

drop_cols = missing[missing > threshold].index.tolist()
print("\nDropping columns with >", threshold, "missing values (count:):", len(drop_cols))
print(drop_cols)
df2 = df.drop(columns=drop_cols).copy()

indicator_cols = []
for c in df2.columns:
    if df2[c].isna().sum() > 0:
        ind_name = c + "_missing"
        df2[ind_name] = df2[c].isna().astype(int)
        indicator_cols.append(ind_name)
print("\nCreated missing-indicator columns for:", len(indicator_cols))

numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df2.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nNumeric cols count:", len(numeric_cols))
print("Categorical cols count:", len(cat_cols))

num_imputer = SimpleImputer(strategy='median')
df_num = pd.DataFrame(num_imputer.fit_transform(df2[numeric_cols]),
                      columns=numeric_cols, index=df2.index)

cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
if len(cat_cols) > 0:
    df_cat = pd.DataFrame(cat_imputer.fit_transform(df2[cat_cols]),
                          columns=cat_cols, index=df2.index)
else:
    df_cat = pd.DataFrame(index=df2.index)

df_clean = pd.concat([df_num, df_cat], axis=1)

print("\nAfter imputation, missing per column (should be 0):")
print(df_clean.isna().sum().sort_values(ascending=False).head(10))

print("\nCleaned dataframe shape:", df_clean.shape)
display(df_clean.head())


# #### 2.1.3. Column correlation

# In this section we will calculate the correlation of different variables with our target variable (sale price) and show a heatmap of the correlations between variables that make the most sense.

# In[4]:


num_df = df.select_dtypes(include=[np.number]).copy()

corr = num_df.corr()

if 'SalePrice' in corr:
    sale_corr = corr['SalePrice'].sort_values(ascending=False)
    print("Top correlations with SalePrice:")
    display(sale_corr.head(30))

selected = ['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Year Built', 'Total Bsmt SF', '1st Flr SF', 'Full Bath']
selected = [c for c in selected if c in num_df.columns]
sns.heatmap(num_df[selected].corr(), annot=True, fmt=".2f", cmap='vlag')
plt.title("Correlation (selected features)")
plt.show()


# As we can see, there are quite a few variables with high correlation with our target variable such as Overall Qual, Gr Liv Area and Garage Cars.
# 
# We can also see that there isn't a very strong correlation between other variables (with a few exceptions such as 1st Flr SF with Total Bsmt SF) which is good as too much correlation would result in a lower performance for our model.

# #### 2.1.4. Scatter plot

# In this section we will plot the scatter plot of Gr Liv Area vs SalePrice to see their relationship, we will also fit a linear regression line to examine a linear relationship and LOWESS to see nonlinearity.

# In[5]:


from statsmodels.nonparametric.smoothers_lowess import lowess

x = 'Gr Liv Area'
y = 'SalePrice'
if x not in df.columns or y not in df.columns:
    raise ValueError(f"Columns {x} or {y} not found in dataframe. Available numeric cols: {list(num_df.columns)[:30]}")

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x=x, y=y, alpha=0.7, edgecolor=None)

mask = df[[x,y]].dropna()
coeffs = np.polyfit(mask[x], mask[y], deg=1)
x_line = np.linspace(mask[x].min(), mask[x].max(), 100)
y_line = np.polyval(coeffs, x_line)
plt.plot(x_line, y_line, linestyle='--', linewidth=2, label=f'Linear fit: y={coeffs[0]:.2f}x + {coeffs[1]:.0f}')


low = lowess(mask[y], mask[x], frac=0.3)
plt.plot(low[:,0], low[:,1], color='orange', linewidth=2, label='LOWESS')


plt.xlabel('GrLivArea (sqft)')
plt.ylabel('SalePrice ($)')
plt.title('Scatter: GrLivArea vs SalePrice')
plt.legend()
plt.show()


# We can see that there is a relation between these two variable but it's clearly not enough to predict the values and we need the rest of the columns too.

# #### 2.1.5. Distribution

# In this section we will plot the histogram and KDE for SalePrice and compute skewness for it.

# In[6]:


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.histplot(df['SalePrice'].dropna(), kde=True)
plt.title('SalePrice distribution')

plt.subplot(1,2,2)
sns.histplot(np.log1p(df['SalePrice'].dropna()), kde=True)
plt.title('log(1 + SalePrice) distribution')

plt.show()

skew_orig = df['SalePrice'].dropna().skew()
skew_log = np.log1p(df['SalePrice'].dropna()).skew()
print(f"Skewness - original SalePrice: {skew_orig:.3f}; log(1+SalePrice): {skew_log:.3f}")


# #### 2.1.6. Outlier check

# In this section we will use the iqr to check for potential outliers.

# In[7]:


def iqr_outliers(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return (series < lower) | (series > upper)

for col in ['Gr Liv Area', 'SalePrice']:
    if col in df.columns:
        mask_out = iqr_outliers(df[col].dropna())
        print(f"{col}: {mask_out.sum()} outliers out of {len(df[col].dropna())}")

outliers_mask = iqr_outliers(df['Gr Liv Area']) | iqr_outliers(df['SalePrice'])
display(df.loc[outliers_mask].sort_values('SalePrice').head(10))


# ### **Step 2: Linear Regression with Scikit-Learn**  

# - Fit a linear regression model using `GrLivArea` to predict `SalePrice`.
# - Print coefficients and intercept.
# - Evaluate model using MAE and R².

# For this section, we will take it one step further. We will train a simple model as described, then we will train a model with all the features and PCA. For the simple model we will also show the coefficients.
# 
# Finally we will evaluate all the models with MAE and R2.

# #### 2.2.1. Simple linear regression (Gr Liv Area only)

# Here we will fit the simple regression version.

# In[8]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df_clean = df.copy()

df = df_clean.copy()

assert 'SalePrice' in df.columns, "df_clean must contain 'SalePrice'."

X_full = df.drop(columns=['SalePrice'])
y_full = df['SalePrice'].values

X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y_full, test_size=0.20, random_state=42)

X_train_simple = X_train_full[['Gr Liv Area']].copy()
X_test_simple  = X_test_full[['Gr Liv Area']].copy()

lr_simple = LinearRegression()
mask_train = X_train_simple['Gr Liv Area'].notna()
mask_test  = X_test_simple['Gr Liv Area'].notna()

lr_simple.fit(X_train_simple[mask_train], y_train[mask_train])

y_pred_simple = lr_simple.predict(X_test_simple[mask_test])

mae_simple = mean_absolute_error(y_test[mask_test], y_pred_simple)
r2_simple  = r2_score(y_test[mask_test], y_pred_simple)

print("SIMPLE MODEL (Gr Liv Area only)")
print("-" * 40)
print(f"Slope (dollars per sqft): {lr_simple.coef_[0]:.2f}")
print(f"Intercept: {lr_simple.intercept_:.2f}")
print(f"Test MAE: {mae_simple:.2f}")
print(f"Test R^2: {r2_simple:.4f}")


# As we can see, the model has been trained successfully but the results aren't ideal (which we saw in the section where we drew the scatter plot and fitted a regression model to it!)

# #### 2.2.2. Regression with more features

# In this section we will build a pipeline that scales the numerical features and one hot encodes the categorical features and performs a PCA on them before fitting a linear regression to it to see if adding other features would result in a better performance or not.

# In[9]:


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'SalePrice']
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()

print("Num numeric cols:", len(numeric_cols))
print("Num categorical cols:", len(cat_cols))

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='drop', verbose_feature_names_out=False)

pca_pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('lr', LinearRegression())
])

pca_pipeline.fit(X_train_full, y_train)

y_pred_pca = pca_pipeline.predict(X_test_full)

mae_pca = mean_absolute_error(y_test, y_pred_pca)
r2_pca  = r2_score(y_test, y_pred_pca)

n_components = pca_pipeline.named_steps['pca'].n_components_
print("\nPCA + LinearRegression model")
print("-" * 40)
print(f"PCA components kept (95% variance): {n_components}")
print(f"Test MAE: {mae_pca:.2f}")
print(f"Test R^2: {r2_pca:.4f}")


# We can clearly see a noticeable performance gain here!

# #### 2.2.3. Comparison

# Here we will compare the results.

# In[ ]:


import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

common_mask = X_test_full['Gr Liv Area'].notna()
y_test_common = y_test[common_mask]
y_pred_simple_common = lr_simple.predict(X_test_simple[common_mask])

y_pred_pca_common = pca_pipeline.predict(X_test_full[common_mask])

results = pd.DataFrame({
    'model': ['Simple (Gr Liv Area)', 'PCA(95%) + Linear'],
    'MAE': [
        mean_absolute_error(y_test_common, y_pred_simple_common),
        mean_absolute_error(y_test_common, y_pred_pca_common)
    ],
    'R2': [
        r2_score(y_test_common, y_pred_simple_common),
        r2_score(y_test_common, y_pred_pca_common)
    ]
})

print("Comparison on test rows with Gr Liv Area present:")
display(results)

median_gr = X_train_full['Gr Liv Area'].median()
X_test_simple_imputed = X_test_full[['Gr Liv Area']].fillna(median_gr)
y_pred_simple_imputed = lr_simple.predict(X_test_simple_imputed)

results_all = pd.DataFrame({
    'model': ['Simple (Gr Liv Area) imputed', 'PCA(95%) + Linear (all test rows)'],
    'MAE': [
        mean_absolute_error(y_test, y_pred_simple_imputed),
        mean_absolute_error(y_test, y_pred_pca)
    ],
    'R2': [
        r2_score(y_test, y_pred_simple_imputed),
        r2_score(y_test, y_pred_pca)
    ]
})
print("\nComparison on full test set (simple model used median imputation for missing Gr Liv Area):")
display(results_all)

best_model_idx = results['MAE'].idxmin()
best_model = results.loc[best_model_idx, 'model']
print(f"\nSummary: On the common test subset, the model with lower MAE is: {best_model}")


# As we can see the model with all the features had a noticeably better performance with a R2 score of 0.87 compared to 0.52

# ### **Step 3: Polynomial Regression**  

# - Use `PolynomialFeatures` to add quadratic terms.
# - Fit and evaluate the polynomial model.
# - Compare performance with linear regression.
# 

# Similar to the last step, we will go one further by training two models, one with only the single feature and one with all the features and PCA (both will use polynomial regression)

# #### 2.3.1. Single feature

# Let's train the polynomial regression on a single feature first

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = df_clean.copy()

X_train_full, X_test_full, y_train, y_test

X_train_simple = X_train_full[['Gr Liv Area']].copy()
X_test_simple  = X_test_full[['Gr Liv Area']].copy()

mask_train = X_train_simple['Gr Liv Area'].notna()
mask_test  = X_test_simple['Gr Liv Area'].notna()

poly = PolynomialFeatures(degree=2, include_bias=True) 
X_train_poly = poly.fit_transform(X_train_simple[mask_train].values.reshape(-1,1))
X_test_poly  = poly.transform(X_test_simple[mask_test].values.reshape(-1,1))

lr_poly_simple = LinearRegression()
lr_poly_simple.fit(X_train_poly, y_train[mask_train])

feat_names = poly.get_feature_names_out(['Gr Liv Area'])

print("Polynomial features (simple) names:", feat_names)

coefs = np.concatenate(([lr_poly_simple.intercept_], lr_poly_simple.coef_[1:])) \
        if lr_poly_simple.coef_.shape[0] == X_train_poly.shape[1] else np.concatenate(([lr_poly_simple.intercept_], lr_poly_simple.coef_))

print("\nSIMPLE POLYNOMIAL MODEL (Gr Liv Area degree=2)")
print("-" * 50)
print(f"Intercept: {lr_poly_simple.intercept_:.2f}")
for name, coef in zip(feat_names[1:], lr_poly_simple.coef_[1:]):  # skip first because intercept is separate
    print(f"Coef for {name}: {coef:.6f}")

y_pred_simple_poly = lr_poly_simple.predict(X_test_poly)
mae_simple_poly = mean_absolute_error(y_test[mask_test], y_pred_simple_poly)
r2_simple_poly = r2_score(y_test[mask_test], y_pred_simple_poly)

print("\nTest metrics (only rows with Gr Liv Area present):")
print(f"MAE (Poly Gr Liv Area): {mae_simple_poly:.2f}")
print(f"R^2  (Poly Gr Liv Area): {r2_simple_poly:.4f}")


# We can see the results are rather similar to the one without polynomial (we will compare all the models in the last section)

# #### 2.3.2. Multiple feature

# Here we will use all the features but we only add teh polynomial term for the numeric values as adding them for the one hot encoded features would result in too many variables and wouldn't really be helpful.
# 
# Another important thing to note here is I originally tried the pipeline with PCA but the results were quite terrible (R2 score of 0.1) and I'm guessing this is because of the very high correlations when we add the x^2 terms, so I have only left the code that doesn't have PCA and performs noticeably better.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'SalePrice']
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()

print("Number of numeric cols to expand:", len(numeric_cols))
print("Number of categorical cols to one-hot:", len(cat_cols))

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, cat_cols)
], remainder='drop', verbose_feature_names_out=False)

poly_pca_pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('lr', LinearRegression())
])

poly_pca_pipeline.fit(X_train_full, y_train)

y_pred_poly_all = poly_pca_pipeline.predict(X_test_full)
mae_poly_all = mean_absolute_error(y_test, y_pred_poly_all)
r2_poly_all = r2_score(y_test, y_pred_poly_all)

print("\nPolynomial (numeric degree=2) + LinearRegression")
print("-" * 60)
print(f"Test MAE: {mae_poly_all:.2f}")
print(f"Test R^2: {r2_poly_all:.4f}")


# Again, we see a noticeable boost over only using a single variable

# #### 2.3.3. Comparison

# Now let's compare all four models

# In[ ]:




