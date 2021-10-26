import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
import math

from feature_engineering import fill_missing_values, drop_column, transform_altitude

# zindi data
#url="https://github.com/jldbc/coffee-quality-database/raw/master/data/robusta_data_cleaned.csv"
df = pd.read_csv('data/Train.csv')
df.head()

# write function to transform features with boxcox
def convert_zeros(x):
    '''
    function to convert zeros to a postive number 
    so that it can be transformed with the boxcox'''
    if x == 0.0:
        return 0.0000001
    else :
        return x

# try to calculate RMSE for the Baseline model:
from sklearn.metrics import mean_squared_error
import math

# Get a very reduced dataframe for the baseline model
# we choose the highly correlated variable school attendance = yes (psa_00) for our baseline model 
df_base = df[['psa_00', 'target']]
df_base.head()

# put this line in the plot:
x = df_base['psa_00']
y = 100 * x - 5
mse = mean_squared_error(df_base['target'], y)

rmse = math.sqrt(mse)

print("Scores for baseline model:")
print("RMSE of baseline model:", round(rmse, 2))
print("Error in % of baseline model:", round((rmse/np.average(y))*100, 2))
print("Mean of the target:", round(np.average(y), 2))
print("Standard deviation of the target:", round(np.std(y), 2))

# drop the non-numerical features
df = df.drop(['ward', 'ADM4_PCODE'], axis=1)

# drop non-percentage features: (total_households, total_individuals, lat, lon, NL and all-Zero values)
df = df.drop(['total_households', 'total_individuals', 'lat', 'lon', 'NL', 'dw_13', 'lan_13', 'dw_12', 'pw_08', 'pw_07'], axis=1)

# Train test split with same random seed
# Defining X and y
features = df.columns.tolist()
features.remove('target')
X = df[features]
y = df.target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=150, shuffle=True)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# train model
lr = LinearRegression()
lr.fit(X_train,y_train)

# predict target values
y_pred = lr.predict(X_test)

# predict target values
y_pred = lr.predict(X_test)

# check error for predictions
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("Scores for the complex model:")
print("r2 score is ", round(score, 2))
print("mean_sqrd_error is:", round(mean_squared_error(y_test, y_pred), 2))
print("root_mean_squared error is:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

# drop features
feature_list = ['pw_00', 'pw_01', 'psa_00', 'psa_01', 'car_00', 'lln_00', 'stv_00', 'pg_00', 'pg_03', 'lan_00', 'lan_01', 'target']
df_final = df[feature_list]

# apply the boxcox transformation on 

for col in feature_list:
    if col != 'target' and  col != 'psa_00' and  col != 'psa_01' and col != 'car_00' and col != 'pg_00':
        #df_final[col] = df_final[col].apply(convert_zeros)
        df_final[col] = boxcox(df_final[col])[0].reshape(-1,1);

# do the cross validation manually
from sklearn.model_selection import KFold

# Using this to test a model on 5 different splits
kf = KFold(n_splits=5, shuffle=False)

ycol = 'target'
in_cols = feature_list[:-1]

scores = []
for train, test in kf.split(df_final):
  lr = LinearRegression()
  lr.fit(df_final[in_cols].iloc[train], df_final[ycol].iloc[train])
  rmse = np.sqrt(mean_squared_error(df_final[ycol].iloc[test], lr.predict(df_final[in_cols].iloc[test])))
  scores.append(rmse)
  print(rmse)

print("Average score in 5-fold CV:", np.mean(scores))

# Train test split with same random seed
# Defining X and y
features = df_final.columns.tolist()
features.remove('target')
X = df_final[features]
y = df_final.target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=150, shuffle=True)

# Grid search for Linear Regression
from sklearn.model_selection import GridSearchCV

# Defining parameter grid (as dictionary)
param_grid = {"fit_intercept" : [True, False],
              "normalize" : [True, False]
             }

# Instantiate gridsearch and define the metric to optimize 
gs = GridSearchCV(LinearRegression(), param_grid, cv=5, verbose=0, n_jobs=-1)

# Fit gridsearch object to data
gs.fit(X_train, y_train)

# Evaluate the model (Multiple Linear Regression) --> Grid search
# Best score
print('Best score:', round(gs.best_score_, 3))

# Best parameters
print('Best parameters:', gs.best_params_)

# Predict
y_pred = gs.predict(X_test)

# predicting the model fit
score = r2_score(y_test, y_pred)
print("Scores for the Final model:")
print("r2 score is ", round(score, 2))
print("mean_sqrd_error is:", round(mean_squared_error(y_test, y_pred), 2))
print("root_mean_squared error is:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

#saving the model
print("Saving model in the model folder")
filename = 'models/linear_regression_model.sav'
pickle.dump(reg, open(filename, 'wb'))