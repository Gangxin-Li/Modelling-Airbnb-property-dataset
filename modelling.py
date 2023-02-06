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

training_data = load_airbnb("Category")
np.random.seed(2)

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

    for params in dict_hyp:
        model = SGDRegressor(**params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        score = sqrt(mean_squared_error(y_test, y_pred))

        if score < best_score:
            best_params = params
            best_score = score
            dict_metric = {"validation_RMSE": score}

    return best_params, dict_metric


def tune_regression_model_hyperparameters(untuned_model, features, labels, dict_hyper):
    """Returns best tuned hyperparameters

    Takes in a model, features, labels and a dictionary of hyperparameters.
    Uses GridSearchCV to find the best hyperparameters and returns the
    best hyperparameters and the best rmse.
    """

    grid_search = GridSearchCV(untuned_model, dict_hyper, cv = 5)
    grid_search.fit(features, labels)
    # print(grid_search.best_params_)
    best_parameters = grid_search.best_params_
    best_rmse = sqrt(abs(grid_search.best_score_))

    return best_parameters, best_rmse


def save_model(folder, model, best_parameters, best_metrics, par_dir):
    """Saves the model into a specified location

    The function takes in the name of the folder to create, the
    model, best parameters and best metrics. Takes the folder argument
    and uses it to create the folder and uses exception statements if
    the folder already exists and overwrites it. Uses .dump() method
    to save the variables as .joblib and .json files. Uses the par_dir
    to put the best_model file in the parent directory of the folder
    argument.
    """
    best_model = {"Best Model": [], "Best Parameters": [], "Best Metrics": []}
    file_path = os.path.join("C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\", par_dir, "best_model.json")

    if folder == "neural_networks":
        subdir1 = "neural_networks"
        subdir2 = par_dir
        full_path = os.path.join("C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\regression\\", subdir1, subdir2)
        
        os.makedirs(full_path, exist_ok=True) 

        with open(os.path.join(full_path, "hyperparameters.json"), "w") as f:
            json.dump(best_parameters, f)
                
        with open(os.path.join(full_path, "metrics.json"), "w") as f:
            json.dump(best_metrics, f)
        

    else:
        try:
            os.makedirs(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}")
            os.mkdir(os.path.dirname(file_path))
            joblib.dump(model, f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\model.joblib")

            with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\hyperparameters.json", "w") as f:
                json.dump(best_parameters, f)
            
            with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\metrics.json", "w") as f:
                json.dump(best_metrics, f)
            
            with open(file_path, "w") as f:
                json.dump(best_model, f)

        except FileExistsError:
            print("Folder or file already exists, will overwrite with new data")
            joblib.dump(model , f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\model.joblib")

            with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\hyperparameters.json", "w") as f:
                json.dump(best_parameters, f)
            
            with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{folder}\metrics.json", "w") as f:
                json.dump(best_metrics, f)

            with open(file_path, "r") as f:
                existing_data = json.load(f)
            
            new_data = {"Best Model": [str(model)], "Best Parameters": [best_parameters], "Best Metrics": [best_metrics]}
            for key, values in new_data.items():
                existing_data[key] += values
            
            with open(file_path, "w") as f:
                json.dump(existing_data, f)
    
    
def evaluate_all_models(model, model_dir, dict_hyper, model_folder):
    """Evaluates the model given a model and dictionary of hyperparameters as the argument

    Gets the best parameters and metrics by calling the 
    tune_regression_model_hyperparameters function. Then calls the
    save_model function. Prints and returns the best parameters and
    metrics.
    """
    performance_dict = tune_classification_model_hyperparameters(model, training_data[0], training_data[1], dict_hyper)
    save_model(model_dir, model, performance_dict["Best Parameters"], performance_dict["validation_accuracy"], model_folder)
    print(performance_dict["Best Parameters"])
    print(performance_dict["validation_accuracy"])

    return performance_dict["Best Parameters"], performance_dict["validation_accuracy"]


def find_best_model(model_filepath):
    """Finds best model from the best_model.json

    Loads in the best_model.json file which contains the metrics 
    for all the models trained. Picks the model which has the 
    best metrics and returns 
    """
    with open(f"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\\models\\{model_filepath}\\best_model.json", "r") as f:
            best_model_data = json.load(f)
    best_model_data_dict = best_model_data
    best_model_index = best_model_data_dict["Best Metrics"].index(max(best_model_data_dict["Best Metrics"]))
    # print(best_model_index)
    best_model = best_model_data_dict["Best Model"][best_model_index]
    best_hyperparameters = best_model_data_dict["Best Parameters"][best_model_index]
    best_metrics = best_model_data_dict["Best Metrics"][best_model_index]

    print(f"The best model is {best_model}")
    print(f"The best hyperparameters are {best_hyperparameters}")
    print(f"The best metrics are {best_metrics}")

    return eval(best_model), best_hyperparameters, best_metrics

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
    model.set_params(**best_hyperparameters)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred, average="micro")
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)

    performance_dict = {"Best Model": str(model),
                        "Best Parameters": best_hyperparameters,
                        "validation_accuracy": accuracy, 
                        "F1 Score": f1, 
                        "Precision": precision,
                        "Recall": recall
    }
    # print(performance_dict)

    return performance_dict 

if __name__ == "__main__":  
    dict_hyper =  {
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5]
}

     #  Change this dictionary to the relevant model hyperparameters       

    # evaluate_all_models(RandomForestRegressor(), dict_hyper) # Change argument for what model you desire
    # best_model , best_hyperparameters, best_metrics = find_best_model()
    # performance_dict = tune_classification_model_hyperparameters(LogisticRegression(), training_data[0], training_data[1], dict_hyper)
    # save_model("models\\classification\\logistic_regression", LogisticRegression(), performance_dict["Best Parameters"], performance_dict["validation_accuracy"], "classification")
    # evaluate_all_models(GradientBoostingClassifier(), "classification\\gradient_boosting", dict_hyper, "classification")
    best_model, best_hyperparameters, best_metrics = find_best_model("classification")
  # %%