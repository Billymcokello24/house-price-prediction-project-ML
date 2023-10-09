#!/usr/bin/env python
# coding: utf-8

# Documentation for this dataset may be found at: http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

# In[3]:


import pandas as pd


columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']

df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', 
                 sep='\t',
                 usecols=columns)

df.head()


# In[69]:


import warnings
warnings.filterwarnings('ignore')


# In[70]:


df.to_csv("ames_housing.csv", index = False)


# In[71]:


get_ipython().system('pip install skimpy')


# In[72]:


from skimpy import skim
skim(df)


# In[73]:


get_ipython().system('pip install summarytools')
from summarytools import dfSummary


# In[76]:


dfSummary(df)


# In[75]:


get_ipython().system('pip install bamboolib')
import bamboolib as bam
bam.enable()


# In[78]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


# Data preprocessing
# Convert 'Central Air' to numerical values (1 for 'Y', 0 for 'N')
df['Central Air'] = df['Central Air'].apply(lambda x: 1 if x == 'Y' else 0)


# In[26]:


# Handle missing values (replace with mean for simplicity)
df.fillna(df.mean(), inplace=True)


# In[27]:


# Define features (X) and target (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']


# In[28]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[29]:


# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[30]:


# KNN regression for different values of K
mse_train = []
mse_test = []
k_values = range(1, 10)

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    mse_train.append(mean_squared_error(y_train, y_pred_train))
    mse_test.append(mean_squared_error(y_test, y_pred_test))



# In[31]:


# Plotting MSE for different values of K
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_train, label='Train MSE')
plt.plot(k_values, mse_test, label='Test MSE')
plt.xlabel('K (Number of Neighbors)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error vs. K for KNN Regression')
plt.legend()
plt.grid(True)
plt.show()



# In[32]:


# Choose the optimal K (minimize test MSE)
optimal_k = k_values[np.argmin(mse_test)]



# In[33]:


# Display metrics for the optimal K
knn_optimal = KNeighborsRegressor(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_train_optimal = knn_optimal.predict(X_train)
y_pred_test_optimal = knn_optimal.predict(X_test)

train_mse_optimal = mean_squared_error(y_train, y_pred_train_optimal)
test_mse_optimal = mean_squared_error(y_test, y_pred_test_optimal)
train_r2_optimal = r2_score(y_train, y_pred_train_optimal)
test_r2_optimal = r2_score(y_test, y_pred_test_optimal)

print('Optimal K:', optimal_k)
print('Train MSE:', train_mse_optimal)
print('Test MSE:', test_mse_optimal)
print('Train R-squared:', train_r2_optimal)
print('Test R-squared:', test_r2_optimal)


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# In[35]:


# Handle missing values 
df.fillna(df.mean(), inplace=True)

# Define features (X) and target (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']


# In[36]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[37]:


# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_train_linear = linear_reg.predict(X_train)
y_pred_test_linear = linear_reg.predict(X_test)


# In[38]:


# Lasso Regression
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)
y_pred_train_lasso = lasso_reg.predict(X_train)
y_pred_test_lasso = lasso_reg.predict(X_test)


# In[39]:


# Ridge Regression
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)
y_pred_train_ridge = ridge_reg.predict(X_train)
y_pred_test_ridge = ridge_reg.predict(X_test)


# In[40]:


# Computing MSE and R-squared for each model
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


# In[56]:


train_mse_linear, train_r2_linear = calculate_metrics(y_train, y_pred_train_linear)
test_mse_linear, test_r2_linear = calculate_metrics(y_test, y_pred_test_linear)

train_mse_lasso, train_r2_lasso = calculate_metrics(y_train, y_pred_train_lasso)
test_mse_lasso, test_r2_lasso = calculate_metrics(y_test, y_pred_test_lasso)

train_mse_ridge, train_r2_ridge = calculate_metrics(y_train, y_pred_train_ridge)
test_mse_ridge, test_r2_ridge = calculate_metrics(y_test, y_pred_test_ridge)


# In[57]:


print('Linear Regression:')
print('Train MSE:', train_mse_linear)
print('Train R-squared:', train_r2_linear)
print('Test MSE:', test_mse_linear)
print('Test R-squared:', test_r2_linear)

print('\nLasso Regression:')
print('Train MSE:', train_mse_lasso)
print('Train R-squared:', train_r2_lasso)
print('Test MSE:', test_mse_lasso)
print('Test R-squared:', test_r2_lasso)

print('\nRidge Regression:')
print('Train MSE:', train_mse_ridge)
print('Train R-squared:', train_r2_ridge)
print('Test MSE:', test_mse_ridge)
print('Test R-squared:', test_r2_ridge)


# ### The case below shows a linear relationship between overall quality of the house and sale price

# In[52]:


# Load the data
columns = ['Overall Qual', 'SalePrice']
url = 'http://jse.amstat.org/v19n3/decock/AmesHousing.txt'
df = pd.read_csv(url, sep='\t', usecols=columns)

# Plotting "Overall Qual" vs "SalePrice"
plt.figure(figsize=(10, 6))
plt.scatter(df['Overall Qual'], df['SalePrice'], alpha=0.5)
plt.xlabel('Overall Qual')
plt.ylabel('SalePrice')
plt.title('Overall Qual vs SalePrice')
plt.grid(True)
plt.show()


# In[53]:


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Extend features by including polynomial features of degree 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_scaled)

# Split the extended data into training and testing sets (70% train, 30% test)
X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)



# In[54]:


# Linear Regression with polynomial features
linear_reg_poly = LinearRegression()
linear_reg_poly.fit(X_poly_train, y_train)
y_pred_train_poly = linear_reg_poly.predict(X_poly_train)
y_pred_test_poly = linear_reg_poly.predict(X_poly_test)

# Lasso Regression with polynomial features
lasso_reg_poly = Lasso()
lasso_reg_poly.fit(X_poly_train, y_train)
y_pred_train_lasso_poly = lasso_reg_poly.predict(X_poly_train)
y_pred_test_lasso_poly = lasso_reg_poly.predict(X_poly_test)

# Ridge Regression with polynomial features
ridge_reg_poly = Ridge()
ridge_reg_poly.fit(X_poly_train, y_train)
y_pred_train_ridge_poly = ridge_reg_poly.predict(X_poly_train)
y_pred_test_ridge_poly = ridge_reg_poly.predict(X_poly_test)



# In[55]:


# Computing MSE and R-squared for each model with polynomial features
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

train_mse_poly, train_r2_poly = calculate_metrics(y_train, y_pred_train_poly)
test_mse_poly, test_r2_poly = calculate_metrics(y_test, y_pred_test_poly)

train_mse_lasso_poly, train_r2_lasso_poly = calculate_metrics(y_train, y_pred_train_lasso_poly)
test_mse_lasso_poly, test_r2_lasso_poly = calculate_metrics(y_test, y_pred_test_lasso_poly)

train_mse_ridge_poly, train_r2_ridge_poly = calculate_metrics(y_train, y_pred_train_ridge_poly)
test_mse_ridge_poly, test_r2_ridge_poly = calculate_metrics(y_test, y_pred_test_ridge_poly)

print('Linear Regression with Polynomial Features (degree=3):')
print('Train MSE:', train_mse_poly)
print('Train R-squared:', train_r2_poly)
print('Test MSE:', test_mse_poly)
print('Test R-squared:', test_r2_poly)

print('\nLasso Regression with Polynomial Features (degree=3):')
print('Train MSE:', train_mse_lasso_poly)
print('Train R-squared:', train_r2_lasso_poly)
print('Test MSE:', test_mse_lasso_poly)
print('Test R-squared:', test_r2_lasso_poly)

print('\nRidge Regression with Polynomial Features (degree=3):')
print('Train MSE:', train_mse_ridge_poly)
print('Train R-squared:', train_r2_ridge_poly)
print('Test MSE:', test_mse_ridge_poly)
print('Test R-squared:', test_r2_ridge_poly)


# ### Did the Performance improve
# Yes.
# Extending the feature set with polynomial features of degree 3 
# improved performance in all the regression ie. linear, Lasso, and Ridge regressions.
# These models better captured complex relationships in the data with the features extendend,
# resulting in reduced Mean Squared Error (MSE) and increased R-squared values, 
# showing a significant improvement predictive accuracy for the housing dataset hence improvement in the perfomance.

# 
