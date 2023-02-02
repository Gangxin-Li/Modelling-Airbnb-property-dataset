from tabular_data import load_airbnb
import numpy as np
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
"""
table,label = load_airbnb()
columns = ['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']
# for col in columns:
#     print(col,':',table[col].unique())
   
table = table[columns]
# table = table[]
X_train, X_test, Y_train, Y_test = train_test_split(table, label, test_size=0.3, random_state=0)
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train,Y_train)
y_hat = model.predict(X_test)
print("RMSE (scikit-learn):", mean_squared_error(Y_test, y_hat, squared=False))
print("R2 (scikit-learn):", r2_score(Y_test, y_hat))


X_test, X_validation, y_test, y_validation = train_test_split(X_test, Y_test, test_size=0.3)
"""
def tune_regression_model_hyperparameters(model,training,hyperparameter):
    clf = GridSearchCV(model, hyperparameter)
    clf.fit(training[0],training[1])
    return clf.best_params_,clf
def custom_tune_regression_model_hyperparameters(model,training,validation,test,hyperparameter):
    best_params,best_model = tune_regression_model_hyperparameters(model,training,hyperparameter)
    print(best_params)
    model.set_params(**best_params)
    model.fit(training[0],training[1])
    predict_train = model.predict(training[0])
    predict_val = model.predict(validation[0])
    predict_test = model.predict(test[0])
    RMSE_training = mean_squared_error(training[1], predict_train, squared=False)
    RMSE_val = mean_squared_error(validation[1], predict_val, squared=False)
    RMSE_test = mean_squared_error(test[1], predict_test, squared=False)
    print("RMSE (training):", RMSE_training)
    print("RMSE (validation):", RMSE_val)
    print("RMSE (test):", RMSE_test)
    metrics = {'predict_train':RMSE_training, 'predict_val':RMSE_val, 'predict_test':RMSE_test}
    print(metrics)
    save_model(best_model,best_params,metrics)

    # best_model = model(learning_rate='invscaling',max_iter=3000, n_iter_no_change=10, penalty='elasticnet')
    # best_model.fit(training[0],training[1])
    # y_hat = best_model.predict(validation[0])
    # print("RMSE (scikit-learn):", mean_squared_error(validation[1], y_hat, squared=False))
    return
def save_model(model,hyperparameter,metrics,model_file='models/regression/linear_regression'):
    if not os.path.exists(model_file):
        os.makedirs(model_file)
        print("Create folder: ",model_file)
    joblib.dump(model, model_file+'/model.joblib')
    with open(model_file+'/hyperparameters.json', 'w') as f:
        json.dump(hyperparameter, f)
    with open(model_file+'/metrics.json', 'w') as f:
        json.dump(metrics, f)
    
if __name__ == "__main__":
    table,label = load_airbnb()
    columns = ['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']
    table = table[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(table, label, test_size=0.3, random_state=0)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, Y_test, test_size=0.3)
    model = SGDRegressor()
    hyperparameter={
        'penalty':['l2','l1','elasticnet'],
        'max_iter':[1000,2000,3000],
        'learning_rate':['invscaling', 'adaptive', 'optimal', 'constant'],
        'n_iter_no_change':[5,10]
    }

    custom_tune_regression_model_hyperparameters(model,(X_train,Y_train),(X_validation,y_validation),(X_test,y_test),hyperparameter)



