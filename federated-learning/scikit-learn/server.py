import flwr as fl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import utils
from sklearn.metrics import mean_squared_error
from typing import Dict

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

def get_evaluate_fn(model: RandomForestRegressor):
    """Return an evaluation function for server-side evaluation."""

    ## CENTRALISED EVALUATION
    test_labels_at_break_df = pd.read_csv('../../TED/CMAPSSData/RUL_FD001.txt', sep=' ', header=None)
    test_data_df = pd.read_csv('../../data_analysis/fd001/fd001-scaled_test.csv', sep=' ')
    test_at_break_df = test_data_df.groupby('ID').last().reset_index()

    ms_used = test_data_df.columns[2:]

    X_test = test_at_break_df[ms_used].values
    y_test = test_labels_at_break_df.values.squeeze()

     # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):

        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        return loss, {"rmse": loss}

    return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    model = RandomForestRegressor()
    # utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )