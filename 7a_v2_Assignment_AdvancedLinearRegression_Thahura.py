#!/usr/bin/env python
# coding: utf-8

# In[946]:


# importing the requisite libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# In[867]:


#train_Assignment_AdvancedRegression
df = pd.read_csv("train_Assignment_AdvancedRegression.csv")


# In[868]:


df.info()


# In[869]:


df.head()


# len(df.Id) # Rows

# In[870]:


len(df.columns) # columns


# In[871]:


df.columns


# ## Exploratory Data Analysis

# - **Check for null values**
#     - if null values percentage is >50%, dropping them
#         - PoolQC          99.52
#         - MiscFeature     96.30
#         - Alley           93.77
#         - Fence           80.75
#         - FireplaceQu     47.26

# In[872]:


df1= df.apply(lambda x :  x.isnull().sum())
df1.head(100)
df2 = df1[df1>0].apply( lambda x : round(x/len(df.Id)*100,2))
df2.sort_values(ascending=False)


# In[873]:


df.drop(columns=['PoolQC' , 'MiscFeature' ,'Alley' ,'Fence' ,  'FireplaceQu'] , inplace = True)
df.head()


# In[874]:


len(df.columns)


# ## Exploratory Data Analysis
# 
# - **Treat the null values**
#     - LotFrontage : fill with 0
#     - GarageType  : fill with NA (No Garage)
#     - GarageYrBlt : fill with 0
#     - GarageFinish: fill with NA (No Garage)
#     - GarageQual  : fill with NA (No Garage)
#     - GarageCond  : fill with NA (No Garage)
#     - BsmtExposure : fill with NA (No Basement)
#     - BsmtFinType2 : fill with NA (No Basement)  
#     - BsmtFinType1 : fill with NA (No Basement)  
#     - BsmtCond     : fill with NA (No Basement)   
#     - BsmtQual    : fill with NA (No Basement)    
#     - MasVnrArea  : fill with 0
#     - MasVnrType  : fill with None
#     - Electrical  : fill with most common value ,ie SBrkr

# In[875]:


df.LotFrontage.isnull().sum()


# In[876]:


df.LotFrontage.head()
df.LotFrontage= df.LotFrontage.fillna(0)
df.LotFrontage.isnull().sum()


# In[877]:


print('Before GarageType ' , df.GarageType.isnull().sum())
df.GarageType.head()
df.GarageType= df.GarageType.fillna('NA')
print('After GarageType ' , df.GarageType.isnull().sum())


# In[878]:


print('Before GarageYrBlt' ,  df.GarageYrBlt.isnull().sum())
df.GarageYrBlt= df.GarageYrBlt.fillna(0)
print("After GarageYrBlt" , df.GarageYrBlt.isnull().sum())
print()
print('Before GarageFinish' ,  df.GarageFinish.isnull().sum())
df.GarageFinish.head()
df.GarageFinish= df.GarageFinish.fillna('NA')
print("After GarageFinish" , df.GarageFinish.isnull().sum())
print()

print('Before GarageQual' ,  df.GarageQual.isnull().sum())
df.GarageQual.head()
df.GarageQual= df.GarageQual.fillna('NA')
print("After GarageQual" , df.GarageQual.isnull().sum())
print()
print('Before GarageCond' ,  df.GarageCond.isnull().sum())
df.GarageFinish.head()
df.GarageCond= df.GarageCond.fillna('NA')
print("After GarageCond" , df.GarageCond.isnull().sum())
print()
print('Before BsmtExposure' ,  df.BsmtExposure.isnull().sum())
df.BsmtExposure.head()
df.BsmtExposure= df.BsmtExposure.fillna('NA')
print("After BsmtExposure" , df.BsmtExposure.isnull().sum())
print()
print('Before BsmtFinType2' ,  df.BsmtFinType2.isnull().sum())
df.BsmtFinType2.head()
df.BsmtFinType2= df.BsmtFinType2.fillna('NA')
print("After BsmtFinType2" , df.BsmtFinType2.isnull().sum())
print()
print('Before BsmtFinType1' ,  df.BsmtFinType1.isnull().sum())
df.BsmtFinType1.head()
df.BsmtFinType1= df.BsmtFinType1.fillna('NA')
print("After BsmtFinType1" , df.BsmtFinType1.isnull().sum())

print()
print('Before BsmtCond' ,  df.BsmtCond.isnull().sum())
df.BsmtCond.head()
df.BsmtCond= df.BsmtCond.fillna('NA')
print("After BsmtCond" , df.BsmtCond.isnull().sum())
print()

print('Before BsmtQual' ,  df.BsmtQual.isnull().sum())
df.BsmtQual.head()
df.BsmtQual= df.BsmtQual.fillna('NA')
print("After BsmtQual" , df.BsmtQual.isnull().sum())
print()
print('Before MasVnrArea' ,  df.MasVnrArea.isnull().sum())
df.MasVnrArea.head()
df.MasVnrArea= df.MasVnrArea.fillna(0)
print("After MasVnrArea" , df.MasVnrArea.isnull().sum())

print()
print('Before MasVnrType' ,  df.MasVnrType.isnull().sum())
df.MasVnrType.head()
df.MasVnrType= df.MasVnrType.fillna('None')
print("After MasVnrType" , df.MasVnrType.isnull().sum())
print()


# -- Fill the null Electrical with the max used value ie, SBrkr

# In[879]:




print('Before Electrical' ,  df.Electrical.isnull().sum())
df.Electrical.value_counts()
df.Electrical.unique()
df.Electrical= df.Electrical.fillna('Empty')


plt.figure(figsize=[9,7])
plt.pie(df.Electrical.value_counts(),labels=df.Electrical.unique(),startangle=90,shadow=True,explode=(0.1, 0.1,0.1,0.1,0.1,0.1), autopct='%1.1f%%')
plt.axis('equal')
plt.show

df.Electrical = df.Electrical.apply( lambda x : x.replace("Empty", "SBrkr") )
print("After Electrical" , df.Electrical.isnull().sum())
print()


# In[880]:


df.info()


# ## Exploratory Data Analysis
# - **Plot the data**
#     - Numeric data :    
#         - MSSubClass
#         - LotFrontage
#         - LotArea
#         - OverallQual
#         - OverallCond
#         - YearBuilt
#         - YearRemodAdd
#         - MasVnrArea
#         - BsmtFinSF1
#         - BsmtFinSF2
#         - BsmtUnfSF
#         - TotalBsmtSF
#         - 1stFlrSF
#         - 2ndFlrSF
#         - LowQualFinSF
#         - GrLivArea
#         - BsmtFullBath
#         - BsmtHalfBath
#         - FullBath
#         - HalfBath
#         - BedroomAbvGr
#         - KitchenAbvGr
#         - TotRmsAbvGrd
#         - Fireplaces
#         - GarageYrBlt
#         - GarageCars
#         - GarageArea
#         - WoodDeckSF
#         - OpenPorchSF
#         - EnclosedPorch
#         - 3SsnPorch
#         - ScreenPorch
#         - PoolArea
#         - MiscVal
#         - MoSold
#         - YrSold
#         
#         - Predictor Variable :  SalePrice
#         - Drop :  Id
#         

# In[881]:


# dropping Id 
df = df.drop(['Id'], axis=1)


# In[882]:




# all numeric (float and int) variables in the dataset
df_numeric = df.select_dtypes(include=['float64', 'int64'])
df_numeric.columns




# In[883]:



df_numeric.head()


# In[884]:


# correlation matrix
cor = df_numeric.corr()


# **Corrlations :**
# 
# SalePrice :
# - high  +ve correlation with  : OverallQual , 'TotalBsmtSF', '1stFlrSF' , GrLivArea ,'GarageCars', 'GarageArea',
# 
# OverallQual :  
# - has lot of high correlations with other factors like garage, total room, livarea, 
# 
# OverallCond :
# - has high correlation with yr built , yr remodelled
#     

# In[885]:



# plotting correlations on a heatmap

# figure size
plt.figure(figsize=(37,37))

# heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()


# ## Outliers Treatment 
# 

# #### -------------------------------------------------------------------------------------------------------------------------
# Ouliers (ie, above 98 quantile)
#    - LotFrontage >120.8  , 30 records dropped
#    - LotArea > 18837.8  , 29 records deleted
#    - MasVnrArea > 632 , 28 records deleted

# In[886]:


sns.pairplot(df, x_vars=['MSSubClass', 'LotFrontage', 'LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# In[887]:


print(np.quantile(df["LotFrontage"], 0.98))
print(df["LotFrontage"].shape)
df_dropped_outliers = df [ (df["LotFrontage"] <= 120.82 )]
df_dropped_outliers["LotFrontage"].shape


# In[888]:


print(np.quantile(df_dropped_outliers["LotArea"], 0.98))
print(df_dropped_outliers["LotArea"].shape)
df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["LotArea"] <= 18837.8 )]
df_dropped_outliers["LotArea"].shape


# In[889]:



print(np.quantile(df_dropped_outliers["MasVnrArea"], 0.98))
print(df_dropped_outliers["MasVnrArea"].shape)
df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["MasVnrArea"] <= 632 )]
df_dropped_outliers["MasVnrArea"].shape


# #### After outlier treatment 

# In[890]:


sns.pairplot(df_dropped_outliers, x_vars=['MSSubClass', 'LotFrontage', 'LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# #### -------------------------------------------------------------------------------------------------------------------------
# Ouliers (ie, above 98 quantile)
#    - BsmtFinSF2 >658.12  , 30 records dropped
#    - BsmtFinSF1 >1442.64   , 30 records dropped
#    - LowQualFinSF > 360 , 6 records deleted
#    - GrLivArea > 2782 , 30 records deleted
#     

# In[891]:


sns.pairplot(df_dropped_outliers, x_vars=['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# In[892]:


print(np.quantile(df_dropped_outliers["BsmtFinSF2"], 0.98))
print(df_dropped_outliers["BsmtFinSF2"].shape)
df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["BsmtFinSF2"] <= 658.12 )]
df_dropped_outliers["BsmtFinSF2"].shape


# In[893]:


print(np.quantile(df_dropped_outliers["BsmtFinSF1"], 0.98))
print(df_dropped_outliers["BsmtFinSF1"].shape)
df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["BsmtFinSF1"] <= 1442.64 )]
df_dropped_outliers["BsmtFinSF1"].shape


# In[894]:


print(np.quantile(df_dropped_outliers["LowQualFinSF"], 0.99))
print(df_dropped_outliers["LowQualFinSF"].shape)

df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["LowQualFinSF"] <= 229 )]
df_dropped_outliers["LowQualFinSF"].shape


# **After outlier treatment**

# In[895]:


sns.pairplot(df_dropped_outliers, x_vars=['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# In[896]:


sns.pairplot(df_dropped_outliers, x_vars=['BsmtFullBath', 'BsmtHalfBath', 
'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# In[897]:


sns.pairplot(df_dropped_outliers, x_vars=['GarageYrBlt', 'GarageCars', 'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# In[898]:


sns.pairplot(df_dropped_outliers, x_vars=['MiscVal', 'MoSold','YrSold'], y_vars='SalePrice',size=3, aspect=0.5, kind='scatter')


# #### -------------------------------------------------------------------------------------------------------------------------
# Ouliers (ie, above 98 quantile)
#    - PoolArea most of the values are 0, can be dropped completely
#   

# In[899]:


print(np.quantile(df_dropped_outliers["PoolArea"], 0.98))
print(df_dropped_outliers["PoolArea"].shape)
#df_dropped_outliers = df [ (df["PoolArea"] <= 2783 )]
#df_dropped_outliers["PoolArea"].shape

# dropping PoolArea 
df_dropped_outliers = df_dropped_outliers.drop(['PoolArea'], axis=1)


# #### -------------------------------------------------------------------------------------------------------------------------
# Ouliers (ie, above 98 quantile)
#    - SalePrice > 394628.49 , 13 records deleted

# In[900]:


print(np.quantile(df_dropped_outliers["SalePrice"], 0.99))
print(df_dropped_outliers["SalePrice"].shape)

df_dropped_outliers = df_dropped_outliers [ (df_dropped_outliers["SalePrice"] <= 394628.49 )]
df_dropped_outliers["SalePrice"].shape


# In[901]:


len(df_dropped_outliers.columns)


# In[902]:


df_dropped_outliers.shape  #(1285, 74)


# In[903]:


df_old=df


# In[904]:


df=df_dropped_outliers


# In[905]:


df.shape  #(1285, 74)


# ## Data Preparation

# In[906]:


# split into X and y
X = df.loc[:, ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',
       'MoSold', 'YrSold']] # predictors in variable X

y = df['SalePrice'] # response variable in Y
len(y)


# In[907]:


X.shape #(1285)


# In[908]:


# creating dummy variables for categorical variables

# subset all categorical variables
df_categorical = df.select_dtypes(include=['object'])
df_categorical.head()


# In[909]:


# convert into dummies - one hot encoding
df_dummies = pd.get_dummies(df_categorical, drop_first=True)
df_dummies.columns


# In[910]:


len(df_dummies)


# In[911]:


# concat dummy variables with X
X = pd.concat([X, df_dummies], axis=1)


# In[912]:


X.shape


# In[913]:


# scaling the features - necessary before using Ridge or Lasso
from sklearn.preprocessing import scale

# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[914]:


print(X.shape)
print(len(y)) #1285, 232


# In[928]:


X.isna().sum()


# In[930]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# ## Model Building and Evaluation

# ### Linear Regression

# In[938]:


# Instantiate
lm = LinearRegression()


# In[939]:


len(X_train.columns)


# In[940]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[944]:


rfe = RFE(lm, n_features_to_select=10)             
rfe = rfe.fit(X_train, y_train)        # running RFE
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[942]:


col = X_train.columns[rfe.support_]
col


# In[947]:


# predict prices of X_test
y_pred = rfe.predict(X_test)

# evaluate the model on test set
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# In[948]:


def calculateVif(X_train) : 
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by='VIF' , ascending=False)
    print(vif)


# In[949]:


calculateVif(X_train)


# In[954]:


# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 40))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)         


# In[955]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[957]:


# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r2')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')


# In[585]:


# Print the coefficients and intercept
print(lm.intercept_)
print(lm.coef_)


# In[958]:


from sklearn.metrics import r2_score, mean_squared_error


# In[959]:


y_pred_train = lm.predict(X_train)
y_pred_test = lm.predict(X_test)

metric = []
r2_train_lr = r2_score(y_train, y_pred_train)
print("r2_train_lr = " , r2_train_lr)
metric.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print("r2_test_lr = ", r2_test_lr)
metric.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print("rss1_train_lr = ", rss1_lr)
metric.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print("rss2_test_lr = ", rss2_lr)
metric.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print("mse_train_lr = ", mse_train_lr)
metric.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print("mse_test_lr = ", mse_test_lr)
metric.append(mse_test_lr**0.5)


# **Error terms are less on train and more on the test**
# 
# **Its an Overfitting model**

# ### Checking for assumptions : Train data

# In[960]:


# Residual analysis
y_res = y_train - y_pred_train # Residuals


# In[961]:


# Residual Analysis
data = pd.DataFrame()
data['res'] = y_res
plt.scatter( y_pred_train , data['res'])
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.show()


# In[962]:


# Distribution of errors
p = sns.distplot(y_res,kde=True)

p = plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# ### Checking for assumptions : test data

# In[963]:


# Residual analysis
y_res = y_test - y_pred_test # Residuals


# In[964]:


# Residual Analysis
data = pd.DataFrame()
data['res'] = y_res
plt.scatter( y_pred_test , data['res'])
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.show()


# In[965]:


# Distribution of errors
p = sns.distplot(y_res,kde=True)

p = plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# In[ ]:





# ## Ridge and Lasso Regression

# In[979]:


# list of alphas to tune - if value too high it will lead to underfitting, if it is too low, 
# it will not handle the overfitting
# params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
#  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
#  4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


params = {'alpha': [10.0, 20, 50, 100, 500, 1000]}


# params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
#  0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
#  4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}

ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error',  
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 
#https://scikit-learn.org/stable/modules/model_evaluation.html


# In[980]:


# Printing the best hyperparameter alpha
print(model_cv.best_params_)


# In[986]:


model_cv.best_params_['alpha']


# In[987]:


cv_results = pd.DataFrame(model_cv.cv_results_)

#Plotting mean test and train scores with alpha
plt.figure(figsize=(16,8))
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('Alpha')
plt.title('neg_mean_absolute_error vs Alpha')
plt.ylabel("neg_mean_absolute_error")
plt.legend(['test score', 'train score'], loc='upper left')
plt.show()


# **Higher NMSE is better. Choosing alpha=100**

# In[ ]:





# In[1028]:


#Fitting Ridge model for alpha = 100 and printing coefficients which have been penalised
alpha = model_cv.best_params_['alpha']


ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
print(ridge.coef_)


# In[1029]:


print(list(zip( X.columns, ridge.coef_)))


# **Most important predictor : 'OverallQual',	6822.341448**
# 

# In[1019]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = ridge.predict(X_train)
y_pred_test = ridge.predict(X_test)

metric2 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print("r2_train_lr = " , r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print("r2_test_lr = " , r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print("rss1_train_lr = " , rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print("rss2_test_lr = " , rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print("mse_train_lr = " , mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print("mse_test_lr = " , mse_test_lr)
metric2.append(mse_test_lr**0.5)


# ## Check for assumptions : Train data 

# In[990]:


# Residual analysis
y_res = y_train - y_pred_train # Residuals


# In[991]:


# Residual Analysis
data = pd.DataFrame()
data['res'] = y_res
plt.scatter( y_pred_train , data['res'])
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.show()


# In[992]:


# Distribution of errors
p = sns.distplot(y_res,kde=True)

p = plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# ## Check for assumptions : Test data

# In[993]:


# Residual analysis
y_res = y_test - y_pred_test # Residuals


# In[994]:


# Residual Analysis
data = pd.DataFrame()
data['res'] = y_res
plt.scatter( y_pred_test , data['res'])
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.show()


# In[995]:


# Distribution of errors
p = sns.distplot(y_res,kde=True)

p = plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# ## Lasso

# In[1000]:


lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 


# In[1003]:


cv_results = pd.DataFrame(model_cv.cv_results_)

#Plotting mean test and train scores with alpha
plt.figure(figsize=(16,8))
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('Alpha')
plt.title('neg_mean_absolute_error  vs Alpha')
plt.ylabel("neg_mean_absolute_error")
plt.show()


# In[1002]:


# Printing the best hyperparameter alpha
print(model_cv.best_params_)


# In[ ]:


model_cv.best_params_['alpha']


# In[1040]:


#Fitting Ridge model for alpha = 100 and printing coefficients which have been penalised

alpha =model_cv.best_params_['alpha']
#alpha=1000
lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 


# In[1041]:


lasso.coef_


# In[1042]:


# Lets calculate some metrics such as R2 score, RSS and RMSE

y_pred_train = lasso.predict(X_train)
y_pred_test = lasso.predict(X_test)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print("r2_train_lr = " , r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print("r2_test_lr = " , r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print("rss1_train_lr = " , rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print("rss2_test_lr = " , rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print("mse_train_lr = " , mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print("mse_test_lr = " , mse_test_lr)
metric3.append(mse_test_lr**0.5)


# In[1043]:


print(list(zip( X.columns, lasso.coef_)))


# In[1044]:


# Residual analysis
y_res = y_train - y_pred_train # Residuals


# In[1045]:


# Residual Analysis
data = pd.DataFrame()
data['res'] = y_res
plt.scatter( y_pred_train , data['res'])
plt.axhline(y=0, color='r', linestyle=':')
plt.xlabel("Predictions")
plt.ylabel("Residual")
plt.show()


# In[1046]:


# Distribution of errors
p = sns.distplot(y_res,kde=True)
p = plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# In[1047]:


# Creating a table which contain all the metrics

lr_table = {'Metric': ['R2 Score (Train)','R2 Score (Test)','RSS (Train)','RSS (Test)',
                       'MSE (Train)','MSE (Test)'], 
        'Linear Regression': metric
        }


# In[1048]:


lr_metric = pd.DataFrame(lr_table ,columns = ['Metric', 'Linear Regression'] )

rg_metric = pd.Series(metric2, name = 'Ridge Regression')
ls_metric = pd.Series(metric3, name = 'Lasso Regression')

final_metric = pd.concat([lr_metric, rg_metric, ls_metric], axis = 1)

final_metric


# ### Changes in the coefficients after regularization

# In[1049]:


betas = pd.DataFrame(index=X.columns)


# In[1050]:


betas.rows = X.columns


# In[1051]:


betas['Linear'] = lm.coef_
betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_


# In[1052]:


pd.set_option('display.max_rows', None)
betas.head(68)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




