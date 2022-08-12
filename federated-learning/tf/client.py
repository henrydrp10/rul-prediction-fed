from pyexpat import model
from tkinter import SINGLE
from xmlrpc import client
import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, Convolution1D, Input, Permute, MaxPool1D , multiply, Concatenate, Flatten
from keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import argparse

parser = argparse.ArgumentParser(description = 'Choose partition number')
parser.add_argument('--partition', type = int, help = 'Partition Number')
args = parser.parse_args()

train_df = pd.read_csv('../FL-data/tf/fd001/scaled/2 workers/90-10/train_partition_' + str(args.partition) + '.csv', sep=',')
test_df = pd.read_csv('../FL-data/tf/fd001/scaled/2 workers/90-10/test_partition_' + str(args.partition) + '.csv', sep=',')

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
            if data_arr.shape[0] - window_length > 0:
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

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(window_length, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def temporal_spatial_fusion_with_attention_model():
    input_data = Input(shape=(window_length, X_train.shape[-1]))
    cnn_layer1 = Convolution1D(64, kernel_size = 3)(input_data)
    cnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer1)
    cnn_layer3 = Convolution1D(32, kernel_size = 3)(cnn_layer2)
    cnn_layer4 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer3)
    cnn_layer5 = Flatten()(cnn_layer4)
    cnn_layer6 = Dense(10, activation = 'relu')(cnn_layer5)
    attention = attention_3d_block(input_data)
    lstm_layer1 = LSTM(128, activation = 'tanh', return_sequences = True)(attention)
    lstm_layer2 = LSTM(64, activation = 'tanh', return_sequences = True)(lstm_layer1)
    lstm_layer3 = LSTM(32, activation = 'tanh', return_sequences = True)(lstm_layer2)
    lstm_layer4 = LSTM(32, activation = 'tanh')(lstm_layer3)
    lstm_layer5 = Dense(10, activation = 'relu')(lstm_layer4)
    merged = Concatenate(axis = 1)([cnn_layer6, lstm_layer5])
    ffnn_layer1 = Dense(128, activation = 'relu')(merged)
    ffnn_layer2 = Dropout(0.2)(ffnn_layer1)
    ffnn_layer3 = Dense(32, activation = 'relu')(ffnn_layer2)
    out = Dense(1)(ffnn_layer3)
    return Model(input_data, out)

client_model = temporal_spatial_fusion_with_attention_model()
client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0008))

# Define Flower client
class TEDClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return client_model.get_weights()

    def fit(self, parameters, config):
        client_model.set_weights(parameters)
        client_model.fit(X_train, y_train, epochs=5, validation_data = (X_val, y_val), batch_size = 64)
        return client_model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        client_model.set_weights(parameters)
        test_cnn_pred = client_model.predict(X_test)
        mse = mean_squared_error(y_test, test_cnn_pred)
        rmse = np.sqrt(mse)
        return rmse, len(y_test), {"rmse": rmse}


# Start Flower client
fl.client.start_numpy_client(server_address = "127.0.0.1:8080", client=TEDClient())