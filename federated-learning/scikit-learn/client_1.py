from os import renames
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

    model = RandomForestRegressor(n_estimators=600,
                                max_features="sqrt", 
                                random_state=42, 
                                max_depth=8, 
                                min_samples_leaf=50)

class TEDClient(fl.client.NumPyClient):
    def get_parameters(self):  
        return model.get_params()

    def fit(self, parameters):
        model.set_params(parameters)
        model.fit(X_train, y_train)
        return model.get_params(), len(X_train), {}

    def evaluate(self, parameters):  
        model.set_params(parameters)
        loss = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

fl.client.start_numpy_client(server_address = "[::]:8080", client=TEDClient())