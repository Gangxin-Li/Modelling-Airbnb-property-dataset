from tabular_data import load_airbnb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
import numpy as np
import itertools
import typing
from os import walk
import joblib
import os
import json

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def tune_classification_model_hyperparameters(model,table,label,grid):
    X_train, X_validation, y_train, y_validation = train_test_split(table, label, test_size=0.2)
    best_hyperparams, best_loss = None, np.inf
    for hyperparams in grid_search(grid):
        model = model.set_params(**hyperparams)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_validation)
        validation_loss = accuracy_score(y_validation, y_validation_pred)

        print(f"H-Params: {hyperparams} Validation loss: {validation_loss}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_hyperparams = hyperparams

    print(f"Best loss: {best_loss}")
    print(f"Best hyperparameters: {best_hyperparams}")
    ## Best model
    best_model = model.set_params(**best_hyperparams)
    best_model.fit(X_train,y_train)
    predict_train = best_model.predict(X_train)
    predict_val = best_model.predict(X_validation)

    f1_training = f1_score(y_train, predict_train,average = 'micro')
    f1_val = f1_score(y_validation, predict_val,average = 'micro')
    acc_training = accuracy_score(y_train, predict_train)
    acc_val = accuracy_score(y_validation, predict_val)
    precision_training = precision_score(y_train, predict_train,average = 'macro')
    precision_val = precision_score(y_validation, predict_val,average = 'macro')
    recall_score_training = recall_score(y_train, predict_train,average = 'macro')
    recall_score_val = recall_score(y_validation, predict_val,average = 'macro')
    print("f1_score (training):", f1_training)
    print("f1_score (validation):", f1_val)
    print("acc_training:", acc_training)
    print("acc_val:", acc_val)
    print("precision_training:", precision_training)
    print("precision_val:", precision_val)
    print("recall_score_training:", recall_score_training)
    print("recall_score_val:", recall_score_val)
    best_metrics = {'f1_score_train':f1_training, 'f1_score_val':f1_val, 'acc_training': f1_training,
    'acc_val:': f1_val,'precision_training':precision_training,'precision_val':precision_val,
    'recall_score_training':recall_score_training,'recall_score_val':recall_score_val}
    return model,best_hyperparams,best_metrics

def save_model(model,hyperparameter,metrics,classification='classification',folder='logistic_regression',root='models'):
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

def evaluate_all_models(models,table,label,grid):
    best_results = []
    for num in range(len(models)):
        best_model,best_hyperparams,best_metrics = tune_classification_model_hyperparameters(models[num],table,label,grid[num])
        best_results.append([best_model,best_hyperparams,best_metrics])
    choose_model = [x[2]['predict_val'] for x in best_results]
    best = np.argmin(choose_model)
    return best_results[best]

def find_best_model(address):
    best_performance = np.inf
    res = []
    for (dirpath, dirnames, filenames) in walk(address):
        for file in filenames:
            # print(dirpath,file)
            
            if file == "metrics.json":
                with open( dirpath+'/'+file , "r" ) as read_content:
                    metric = json.load(read_content)
            if metric['predict_val']<best_performance:
                best_performance = metric['predict_val']
                model = joblib.load(dirpath+'/model.joblib')
                with open( dirpath+'/hyperparameters.json' , "r" ) as read_content:
                    hyperparameters = json.load(read_content)
    
                res = [model,hyperparameters,metric]
    # print(res)
    return res

if __name__ == '__main__':
    table,label = load_airbnb('Category')
    print(table.columns)
    print(label.unique())
    columns = ['guests','beds','bathrooms','Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating','amenities_count','bedrooms']
    table = table[columns]
    max_abs_scaler = preprocessing.MaxAbsScaler()
    table = max_abs_scaler.fit_transform(table)
    model = LogisticRegression()
    grid={
        'penalty': ['l2', None],
        'max_iter':[5000,10000,20000],
    }
    best_model,best_hyperparams,best_metrics=tune_classification_model_hyperparameters(model,table,label,grid)
    save_model(best_model,best_hyperparams,best_metrics,classification = 'classification',folder = 'logistic_regression')






