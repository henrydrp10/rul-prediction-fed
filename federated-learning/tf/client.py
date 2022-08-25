from tkinter import SINGLE
from xmlrpc import client
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Convolution1D, Input, Permute, MaxPool1D , multiply, Concatenate, Flatten
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser(description = 'Choose partition number')
parser.add_argument('--partition', type = int, help = 'Partition Number')
args = parser.parse_args()

train_df = pd.read_csv('../FL-data/tf/fd001/scaled/2 workers/50-50/train_partition_' + str(args.partition) + '.csv', sep=',')
test_df = pd.read_csv('../FL-data/tf/fd001/scaled/2 workers/50-50/test_partition_' + str(args.partition) + '.csv', sep=',')

train_labels_df = pd.DataFrame(train_df.pop('RUL')) 
test_labels_df = pd.DataFrame(test_df.pop('RUL'))

def get_windows(data_df, labels_df, window_length, mode = 'train'):

    if mode == 'train':

        labels_df['ID'] = data_df['ID']

        data_groupby = data_df.groupby('ID', sort=False)
        labels_groupby = labels_df.groupby('ID', sort=False)

        id_list = [num[0] for num in data_groupby['ID'].unique()]
        val_indices = np.random.choice(id_list, size = int(0.2 * len(id_list)), replace = False)
        
        tr_data_eng_arr = []
        tr_labels_eng_arr = []

        val_data_eng_arr = []
        val_labels_eng_arr = []

        for i in id_list:
            if i in val_indices:
                val_data_eng_arr.append(data_groupby.get_group(i))
            else:
                tr_data_eng_arr.append(data_groupby.get_group(i))

        for i in id_list:
            if i in val_indices:
                val_labels_eng_arr.append(labels_groupby.get_group(i))
            else:
                tr_labels_eng_arr.append(labels_groupby.get_group(i))

        tr_data_windows = []
        tr_label_windows = []
        for index in range(len(tr_data_eng_arr)):
            tr_data_arr = tr_data_eng_arr[index].to_numpy()
            tr_labels_arr = tr_labels_eng_arr[index].to_numpy()
            if tr_data_arr.shape[0] - window_length + 1 > 0:
                for t in range(tr_data_arr.shape[0] - window_length + 1):
                    tr_data_windows.append(tr_data_arr[t:t+window_length, :])
                    tr_label_windows.append(tr_labels_arr[t+window_length - 1, 0])

        val_data_windows = []
        val_label_windows = []
        for index in range(len(val_data_eng_arr)):
            val_data_arr = val_data_eng_arr[index].to_numpy()
            val_labels_arr = val_labels_eng_arr[index].to_numpy()
            if val_data_arr.shape[0] - window_length + 1 > 0:
                for t in range(val_data_arr.shape[0] - window_length + 1):
                    val_data_windows.append(val_data_arr[t:t+window_length, :])
                    val_label_windows.append(val_labels_arr[t+window_length - 1, 0])

        return np.array(tr_data_windows), np.array(tr_label_windows), np.array(val_data_windows), np.array(val_label_windows)

    else:

        labels_df['ID'] = data_df['ID']

        data_groupby = data_df.groupby('ID', sort=False)
        labels_groupby = labels_df.groupby('ID', sort=False)

        id_list = [num[0] for num in data_groupby['ID'].unique()]

        data_eng_arr = []
        labels_eng_arr = []

        for i in id_list:
            data_eng_arr.append(data_groupby.get_group(i))

        for i in id_list:
            labels_eng_arr.append(labels_groupby.get_group(i))

        data_windows = []
        label_windows = []
        for index in range(len(data_eng_arr)):
            data_arr = data_eng_arr[index].to_numpy()
            labels_arr = labels_eng_arr[index].to_numpy()
            if data_arr.shape[0] - window_length + 1 > 0:
                data_windows.append(data_arr[-window_length:, :])
                label_windows.append(labels_arr[-1, 0])

        return np.array(data_windows), np.array(label_windows)

############ MODEL #######################################################

window_length = 30
X_train, y_train, X_val, y_val = get_windows(train_df, train_labels_df, window_length, mode='train')
X_test, y_test = get_windows(test_df, test_labels_df, window_length, mode = 'test')

y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)
y_test = np.expand_dims(y_test, axis=1)

X_train = X_train[:,:,2:]
X_val = X_val[:,:,2:]
X_test = X_test[:,:,2:]

def mean_and_polynomial_fitting(data_set):
    mean_trends = []
    for window in data_set:
        sensor_means = np.mean(window, axis = 0)
        coefs = []
        for i in range(window.shape[-1]):
            coefs.append(np.polyfit(range(window.shape[0]), window[:,i], 1)[0])
        mean_trend = np.concatenate((sensor_means, np.array(coefs)), axis = 0)
        mean_trends.append(mean_trend)
    return np.array(mean_trends)

# train_extracted = mean_and_polynomial_fitting(X_train)
# val_extracted = mean_and_polynomial_fitting(X_val)
# test_extracted = mean_and_polynomial_fitting(X_test)

# train_extracted = np.array(train_extracted)
# val_extracted = np.array(val_extracted)
# test_extracted = np.array(test_extracted)

# scale = preprocessing.MinMaxScaler()
# train_extracted = scale.fit_transform(train_extracted)
# val_extracted = scale.fit_transform(val_extracted)
# test_extracted = scale.fit_transform(test_extracted)

X_train = mean_and_polynomial_fitting(X_train)
X_val = mean_and_polynomial_fitting(X_val)
X_test = mean_and_polynomial_fitting(X_test)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

scale = preprocessing.MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_val = scale.fit_transform(X_val)
X_test = scale.fit_transform(X_test)


client_model = Sequential()
client_model.add(Dense(64, activation = 'relu', input_dim = X_train.shape[-1]))
client_model.add(Dropout(0.5))
client_model.add(Dense(128, activation = 'relu'))
client_model.add(Dropout(0.5))
client_model.add(Dense(256, activation = 'relu'))
client_model.add(Dense(1))

client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.001))

# def trend_lstm():
#     input_data = Input(shape=(window_length, X_train.shape[-1]))
#     input_extracted = Input(shape=(train_extracted.shape[-1],))
#     mlp_layer1 = Dense(64, activation = 'relu')(input_extracted)
#     mlp_layer2 = Dropout(0.25)(mlp_layer1)
#     mlp_layer3 = Dense(10, activation = 'relu')(mlp_layer2)
#     lstm_layer1 = LSTM(64, activation = 'tanh', return_sequences = True)(input_data)
#     lstm_layer2 = LSTM(32, activation = 'tanh')(lstm_layer1)
#     lstm_layer3 = Dense(10, activation = 'relu')(lstm_layer2)
#     merged = Concatenate(axis = 1)([mlp_layer3, lstm_layer3])
#     ffnn_layer1 = Dense(32, activation = 'relu')(merged)
#     ffnn_layer2 = Dropout(0.25)(ffnn_layer1)
#     out = Dense(1)(ffnn_layer2)
#     return Model([input_data, input_extracted], out)

# client_model = trend_lstm()
# client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0005))

# Define Flower client
class TEDClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return client_model.get_weights()

    def fit(self, parameters, config):
        client_model.set_weights(parameters)
        client_model.fit(X_train, y_train, epochs=5, validation_data = (X_val, y_val), batch_size = 256)
        return client_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        client_model.set_weights(parameters)
        test_cnn_pred = client_model.predict(X_test)
        mse = mean_squared_error(y_test, test_cnn_pred)
        rmse = np.sqrt(mse)
        return rmse, len(y_test), {"rmse": rmse}

# Start Flower client
fl.client.start_numpy_client(server_address = "127.0.0.1:8080", client=TEDClient())