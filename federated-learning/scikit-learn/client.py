import flwr as fl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import utils

def get_data():

    train_data_df = pd.read_csv('../../data_analysis/fd001/fd001-scaled_train.csv', sep=' ')
    test_data_df = pd.read_csv('../../data_analysis/fd001/fd001-scaled_test.csv', sep=' ')

    train_labels_df = pd.read_csv('../../data_analysis/fd001/fd001-training_labels.csv', sep=' ')
    test_labels_df = pd.read_csv('../../data_analysis/fd001/fd001-testing_labels.csv', sep=' ')
    test_labels_at_break_df = pd.read_csv('../../TED/CMAPSSData/RUL_FD001.txt', sep=' ', header=None)
    test_at_break_df = test_data_df.groupby('ID').last().reset_index()

    train_labels_df = train_labels_df.clip(upper = 125)
    ms_used = train_data_df.columns[2:]

    X_train = train_data_df[ms_used].values
    X_test = test_at_break_df[ms_used].values
    y_train = train_labels_df.values.squeeze()
    y_test = test_labels_at_break_df.values.squeeze()

    return (X_train, y_train), (X_test, y_test) 

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = get_data()

    # Model
    model = RandomForestRegressor(
        n_estimators=100, 
        max_features="sqrt", 
        random_state=42, 
        max_depth=8, 
        min_samples_leaf=50
    )

    # utils.set_initial_params(model)

# Flower Client
class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): 
        utils.set_model_params(model, parameters)
        model.fit(X_train, y_train)
        print(f"Training finished for round {config['server_round']}")
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        utils.set_model_params(model, parameters)
        loss = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        return loss, len(X_test), {"rmse": loss}

# Start Flower client
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MnistClient())