

#Predicting the target variable

#Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import cross_val_score

#Creating X and y datasets from the file
data_set = pd.read_csv(r'C:\Users\PraveenT\Desktop\Data Scientist Test\dataset_00_with_header.csv')
X = data_set.drop(['y'], axis=1)
y = data_set['y']


#Exploratory data analysis

# X.corr()  -> to identify the correlations between the target variable and independent variables
# X.info(), X.decsribe() -> to view basic statistics about the distributions of variables
# X.hist()  -> to view histograms of the variables. Many variables seems to be skewed

############################################
#   Eliminating the variables with very low correlation with the target variable
#from sklearn.ensemble import ExtraTreesRegressor
#extree_reg = ExtraTreesRegressor()
#extree_reg.fit(X, y) 
#y_pred = extree_reg.predict(y)
#feature_importance = extree_reg.feature_importances_
#normalized_feature_importance = np.std([tree.feature_importances_ for tree in extree_reg.estimators_], axis = 0)
#feat_and_col = np.concatenate(col_names,normalized_feature_importance)
#normalized_feature_importance = normalized_feature_importance[::-1]
#features_score = pd.DataFrame(feature_importance_normalized, index=col_names)
#features_score.sort_values([frame1.columns[0]],inplace=True, ascending=False)
#features_score.columns = ['imp_score']
#list_Var = features_score[features_score['imp_score']<0.00002]
#list_Var = list_Var.index.tolist()
#X = X.drop(list_Var, axis=1)
#############################################

# Spliting the dataset and scaling the variables
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 0)

X = X.fillna(X.median())
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

##############################################

# Developing and fitting the xgboost model. Other models were tried but xgboost gives very low RMSE

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt

xgbmodel = xgb.XGBRegressor(learning_rate=0.1, max_depth=7, colsample_bytree =0.7, subsample =0.7)
xgbmodel.fit(X_train, y_train)
y_pred = xgbmodel.predict(X_test)
rmse_param = sqrt(mean_squared_error(y_test, y_pred))
# RMSE is ~ 21
################################################

# Grid search to find the best parameters after tuning
#from sklearn.model_selection import GridSearchCV
#param_grid = {'learning_rate': [.03, 0.05, .1], 
#              'max_depth': [5, 6, 7],
#              'subsample': [0.7],
#              'colsample_bytree': [0.7]
 #             }

#xgb_grid = GridSearchCV(estimator=xgbmodel, param_grid=param_grid, cv= 5)
#xgb_grid.fit(X_test, y_test)
#best_scr = xgb_grid.best_score_
#best_params = xgb_grid.best_params_
#print(xgb_grid.best_score_)
#print (xgb_grid.best_params_)
 
 #################################################
