# Airbnb Property dataset

> Tech: AWS, cv2, data cleaning, data modelling

### Task1 Load in the Tabular Dataset

Tech: AWS, Pandas 
Steps:
Download data from AWS
Load the dataset
Remove the missing things
raw text data processing
clean tabular data

### Task2 Load/process img Dataset

Tech: cv2, AWS
Steps:
Download data from AWS
resize image data
convert into RG

## Milestone 1 Data Preparation

```python
def resize_images(address,target_address):
    if not os.path.exists(target_address):
        os.makedirs(target_address)
        print("Create default address:",target_address)
    # shape = []
    for (dirpath, dirnames, filenames) in walk(address):
        for file in filenames:
            # print(dirpath,dirnames,file)
            img = cv2.imread(dirpath+'/'+file)
            img = cv2.resize(img,(540,720),interpolation = cv2.INTER_AREA)
            # print(target_address+'/'+dirnames[0]+'/'+file)
            path = target_address+'/'+dirnames[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
                print("Create folder: ",path)
            cv2.imwrite(path+'/'+file, img)

```

## Milestone 2 Create Regression model

- Create a simple regression model as a base line

- Evaulate the model's performance

- Tune hyperparmeters

- Save the model

- Build Logistic regression model

```python
class LinearRegression(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module("input_layer", torch.nn.Linear(9, config["hidden_layer_width"]))
        self.layers.add_module("activation_function", torch.nn.ReLU())
        for i in range(config["model_depth"] - 2):
            self.layers.add_module(f"hidden_layer_width{i}", torch.nn.Linear(config["hidden_layer_width"], config["hidden_layer_width"]))
            self.layers.add_module("Relu", torch.nn.ReLU())
        self.layers.add_module("output_layer", torch.nn.Linear(config["hidden_layer_width"], 1))
        
    def forward(self, features):
        return self.layers(features)
```

- Train the model

```python
def train(model, dataloader, epoch, config):
    """Training loop for the neural network

    Chooses the optimiser to use based on the randomly generated
    config. Trains model iteratively by the number given for the
    epoch and 
    """
    start_time = time.time()
    dt_now = datetime.now()
    optimiser_name = config["optimiser"]
    
    if optimiser_name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif optimiser_name == "Adadelta":
        optimiser = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Optimiser: {optimiser_name} not supported.")
    batch_index = 0
    writer = SummaryWriter()
    prediction_list = []
    labels_list = []
    num_predictions = 0
    avg_rmse = 0
    for epoch in range(epoch):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            prediction_list.append(prediction)
            labels_list.append(labels.detach().numpy())
            labels = labels.to(prediction.dtype)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            avg_rmse += torch.sqrt(loss)
            print(f"The MSE Loss: {loss.item()}")
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar("loss", loss.item(), batch_index)
            batch_index += 1

    num_predictions += prediction.shape[0]

    labels = np.concatenate(labels_list)
    prediction_list = np.concatenate([pred.detach().numpy() for pred in prediction_list])    
    r2 = r2_score(labels, prediction_list)

    end_time = time.time()
    total_time = end_time - start_time
    dt_string = dt_now.strftime("%d_%m_%Y_%H-%M")
    inference_latency = total_time / num_predictions
    avg_rmse = avg_rmse / num_predictions
    best_metrics = {
        "Avg RMSE_loss": str(avg_rmse), 
        "R_squared": r2, 
        "training_duration": total_time,
        "inference_latency": inference_latency
    }
    
    print(best_metrics)
    return best_metrics, dt_string

```

- Load parameters from ymal and tunning it

```python
def get_nn_config():
    """Loads config .yaml file from directory

    """
    with open(r"C:\\Users\\denni\\Desktop\\AiCore\\Projects\\modelling-airbnbs-property-listing-dataset-\nn_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    return config

def generate_nn_configs():
    """Randomly generates configs based on set list of values

    Creates lists for the optimisers and different values for the
    learning rate, hidden layer width and model depth. Then uses 
    the random.choice to randomly select values for each key then
    creates 16 different configs and returns them.
    """
    optimiser = ["Adam", "Adagrad", "Adadelta"]
    learning_rate = [0.01, 0.001, 0.0001]
    hidden_layer_width = [32, 64, 128, 256]
    model_depth = [9, 10, 11, 12, 13, 14]
    config_list = []

    for index in range(0,17):
        
        config_file = {
        "optimiser": random.choice(optimiser),
        "learning_rate": random.choice(learning_rate),
        "hidden_layer_width": random.choice(hidden_layer_width),
        "model_depth": random.choice(model_depth)
    }
        config_list.append(config_file)
    print(config_list)
    return config_list

def find_best_nn(config_list):
    """Finds best neural network from the different configs

    Iterates and trains the model through the list of configs.
    Initialises best model/r2/config/metric_dict then updates 
    them as it is iterating over the list of configs. The function
    then saves the best model into the specific directory and 
    returns the best config and metric dictionary amongst other
    variables.
    """
    best_r2 = -float("inf")
    best_model = None
    best_config = None
    best_metrics_dict = None
    for config in config_list:
        model = LinearRegression(config)
        best_metrics, dt_string = train(model, train_loader, 25, config)    

        if best_metrics["R_squared"] > best_r2:
            best_r2 = best_metrics["R_squared"]
            best_model = model
            best_config = config
            best_metrics_dict = best_metrics

    os.mkdir(f"Path\{dt_string}")
    torch.save(best_model.state_dict, f"path\{dt_string}\model.pt")
    return best_config, best_metrics_dict, dt_string, best_model

```


## Milestone 2 Tune hyperparmeters

- Custome tune hyerparmeters

```python
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
```

- Tune the regression model

```python
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
```


- Tune the Classification model

```python
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
```


- save the model

```python
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
    
```


- Find the best model

```python
def find_best_model(address):
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
```



## Conclusions

- Maybe write a conclusion to the project, what you understood about it and also how you would improve it or take it further.

- Read through your documentation, do you understand everything you've written? Is everything clear and cohesive?