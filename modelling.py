# %%
from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.model_selection import GridSearchCV
import os
import json
import joblib
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import itertools
import typing

np.random.seed(2)

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def custome_tune_regression_model_hyperparameters(model, features, label, dict_hyp):
    """Finds best hyperparameters without using GridSearchCV

    Takes in the model, features, labels and dictionary of hyperparameter
    values as the arguments. Using a for loop the function iterates
    over the dictionary of hyperparameters and if the current score
    is less than the best score it will replace it until it reaches
    the best score. Returns the best parameters and a dictionary 
    containing the rmse.
    """
    best_params = None
    best_score = float("inf")
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)
    
    for params in grid_search(dict_hyp):
        model = SGDRegressor(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        RMSE_score = sqrt(mean_squared_error(y_test, y_pred))
        R2_score = r2_score(y_test,y_pred)
        if RMSE_score < best_score:
            best_params = params
            best_score = RMSE_score
            best_metric = {"validation_RMSE": RMSE_score, "validation_R2":R2_score}
            best_model = model
    # print(best_params,dict_metric)
    
    return best_model, best_params, best_metric


def tune_regression_model_hyperparameters(untuned_model, features, labels, dict_hyper):
    """Returns best tuned hyperparameters

    Takes in a model, features, labels and a dictionary of hyperparameters.
    Uses GridSearchCV to find the best hyperparameters and returns the
    best hyperparameters and the best rmse.
    """
    grid_search = GridSearchCV(untuned_model, dict_hyper, cv=5)
    grid_search.fit(features, labels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    best_hyperparameters = grid_search.best_params_
    best_model = untuned_model.set_params(**best_hyperparameters)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    RMSE_score = sqrt(mean_squared_error(y_test, y_pred))
    R2_score = r2_score(y_test,y_pred)
    best_metric = {"validation_RMSE": RMSE_score, "validation_R2":R2_score}
    return best_model,best_hyperparameters, best_metric


def save_model(model,hyperparameter,metrics,classification='classification',folder='logistic_regression',root='./airbnb-property-listings/models'):
    """Saved the best model experiement

    Takes in a model, hyperparameter,metrics, and target classification and foler.
    Using joblib to save the model
    Using json.dump to save the parameters.
    """
    
    path = root+'/'+classification+'/'+folder
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create folder: ",path)
    else:
        print("Save the file to: ",path)
    joblib.dump(model, path+'/model.joblib')
    with open(path+'/hyperparameters.json', 'w') as f:
        json.dump(hyperparameter, f)
    with open(path+'/metrics.json', 'w') as f:
        json.dump(metrics, f) 
    

def find_best_model(address):
    """find the best model
    
    for-each models we built, and using theri metrics to evaulate which
    model has the best performacne. and return this model.
    """
    best_performance = np.inf
    res = []
    for (dirpath, dirnames, filenames) in os.walk(address):
        # print(dirpath,dirnames,filenames)
        for file in filenames:
            print(dirpath,file)
            
            if file == "metrics.json":
                with open( dirpath+'/'+file , "r" ) as read_content:
                    metric = json.load(read_content)
            if metric['validation_RMSE']<best_performance:
                best_performance = metric['validation_RMSE']
                model = joblib.load(dirpath+'/model.joblib')
                with open( dirpath+'/hyperparameters.json' , "r" ) as read_content:
                    hyperparameters = json.load(read_content)
    
                res = [model,hyperparameters,metric]
    # print(res)
    return res

def tune_classification_model_hyperparameters(model, features, labels, dict_hyper):
    """Tunes the hyperparameters for classification models

    Splits data into training and testing sets and uses 
    GridSearchCV to carry out the tuning process. It also
    calculates the accuracy F1 score, precision and recall metrics.
    The function then returns a dictionary containing these values
    and variables.
    """
    grid_search = GridSearchCV(model, dict_hyper, cv=5)
    grid_search.fit(features, labels)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    best_hyperparameters = grid_search.best_params_
    best_model = model.set_params(**best_hyperparameters)
    best_model.fit(x_train, y_train)
    
    y_pred = best_model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="micro")
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    best_metrics = {"Best Model": str(model),
                        "validation_accuracy": accuracy, 
                        "F1 Score": f1, 
                        "Precision": precision,
                        "Recall": recall
    }
    
    print(best_metrics)
    return best_model,best_hyperparameters,best_metrics 

if __name__ == "__main__":  
    
    ## Regression
    # table,labels  = load_airbnb("Price_Night")
    # columns = ['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']
    # table = table[columns]

    # SGDRegressor
    # dict_SGDRegressor =  {
    # 'random_state':[2,4,6],
    # 'max_iter':[9000,12000,15000]
    # }
    # best_model,best_hyperparameters,best_metrics = custome_tune_regression_model_hyperparameters(SGDRegressor(),table,labels,dict_SGDRegressor)
    # save_model(best_model,best_hyperparameters,best_metrics,classification='Regression',folder='SGDRegressor')

    # LogisticRegression
    # dict_LogisticRegression =  {
    # 'penalty': ['l2', None],
    # 'max_iter':[5000,10000,20000],
    # }
    # best_model,best_hyperparameters,best_metrics = custome_tune_regression_model_hyperparameters(LogisticRegression(),table,labels,dict_LogisticRegression)
    # save_model(best_model,best_hyperparameters,best_metrics,classification='Regression',folder='LogisticRegression')
    
    # Find the best model
    # find_best_model('./airbnb-property-listings/models/Regression')

    ## Classification
    table,labels  = load_airbnb("Category")
    columns = ['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']
    table = table[columns]

    # LogisticRegression
    # dict_LogisticRegression =  {
    # 'C': [50, 100, 200],
    # 'random_state':[2,4,6],
    # 'max_iter':[9000,12000,15000]
    # }
    # best_model,best_hyperparameters,best_metrics  = tune_classification_model_hyperparameters(LogisticRegression(), table, labels, dict_LogisticRegression)
    # save_model(best_model,best_hyperparameters,best_metrics,classification='Classification',folder='logistic_regression')
    
    # DecisionTreeClassifier
    # dict_DecisionTreeClassifier =  {
    # 'max_leaf_nodes':[500,1000,1200],
    # 'max_depth':[500,800,1000],
    # 'random_state':[2,4,6],
    # }
    # best_model,best_hyperparameters,best_metrics  = tune_classification_model_hyperparameters(DecisionTreeClassifier(), table, labels, dict_DecisionTreeClassifier)
    # save_model(best_model,best_hyperparameters,best_metrics,classification='Classification',folder='DecisionTreeClassifier')

    # gradient_boosting
    dict_gradient_boosting =  {
    'max_leaf_nodes':[500,1000,1200],
    'max_depth':[500,800,1000],
    'random_state':[2,4,6],
    }
    best_model,best_hyperparameters,best_metrics = tune_classification_model_hyperparameters(GradientBoostingClassifier(), table,labels,dict_gradient_boosting)
    save_model(best_model,best_hyperparameters,best_metrics,classification='Classification',folder='gradient_boosting')
# %%