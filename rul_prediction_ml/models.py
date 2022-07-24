from turtle import window_width
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Convolution1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout, Bidirectional
import keras_tuner as kt

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train_data_full_df = pd.read_csv('../data_analysis/fd001-scaled_train_data.csv', sep=' ')
test_data_df = pd.read_csv('../data_analysis/fd001-scaled_test_data.csv', sep=' ')

train_labels_full_df = pd.read_csv('../data_analysis/fd001-training_labels.csv', sep=' ')
test_labels_df = pd.read_csv('../data_analysis/fd001-testing_labels.csv', sep=' ')
test_labels_at_break_df = pd.read_csv('../TED/CMAPSSData/RUL_FD001.txt', sep=' ', header=None)

train_full_df = train_data_full_df.copy()
test_df = test_data_df.copy()
train_labels_full_df = train_labels_full_df.copy().clip(upper=125)
test_labels_df = test_labels_df.copy()

used_sensors = []
used_sensors.append("ID")
used_sensors.append("Cycle")
for i in range(1, 22):
    if i not in [1, 5, 6, 10, 16, 18, 19]:
        used_sensors.append("SensorMeasure" + str(i))

train_full_df = train_full_df[used_sensors]
test_df = test_df[used_sensors]

# Processed data - Numpy
train_full = train_full_df.values
test = test_df.values
train_labels_full = train_labels_full_df.values.squeeze()
test_labels = test_labels_df.values.squeeze()

joined_train_rul = train_full_df.copy()
joined_train_rul['RUL'] = train_labels_full_df['RUL']
test_at_break_df = test_df.groupby('ID').last().reset_index()
test_at_break = test_at_break_df.values

train_labels_at_break = joined_train_rul.groupby('ID').last().reset_index()['RUL'].values
test_labels_at_break = test_labels_at_break_df.values[:, 0]

train_groupby_full_df = train_full_df.groupby('ID', sort=False)
test_groupby_df = test_df.groupby('ID', sort=False)

train_labels_full_df['ID'] = joined_train_rul['ID']
train_labels_full_groupby_df = train_labels_full_df.groupby('ID', sort=False)

val_indices = np.random.choice(len(train_groupby_full_df), size = int(0.2 * len(train_groupby_full_df)))

val_arr = []
train_set_arr = []
val_labels_arr = []
train_set_labels_arr = []

for i in range(len(train_groupby_full_df)):
    if i in val_indices:
        val_arr.append(train_groupby_full_df.get_group(i+1))
        val_labels_arr.append(train_labels_full_groupby_df.get_group(i+1)['RUL'])
    else:
        train_set_arr.append(train_groupby_full_df.get_group(i+1))
        train_set_labels_arr.append(train_labels_full_groupby_df.get_group(i+1)['RUL'])

val_set_df = val_arr[0]
val_labels_df = val_labels_arr[0]
for i in range(1, len(val_arr)):
    val_set_df = pd.concat([val_set_df, val_arr[i]])
    val_labels_df = pd.concat([val_labels_df, val_labels_arr[i]])

train_set_df = train_set_arr[0]
train_set_labels_df = train_set_labels_arr[0]
for i in range(1, len(train_set_arr)):
    train_set_df = pd.concat([train_set_df, train_set_arr[i]])
    train_set_labels_df = pd.concat([train_set_labels_df, train_set_labels_arr[i]])

train_set = train_set_df.values
train_set_labels = train_set_labels_df.values
val_set = val_set_df.values
val_labels = val_labels_df.values
val_labels = np.expand_dims(val_labels, axis = 1)
train_set_labels = np.expand_dims(train_set_labels, axis = 1)
train_labels_full = np.expand_dims(train_labels_full, axis = 1)

def get_windows(data_df, labels_df, window_length, mode = 'train'):

    if mode == 'train':

        labels_df['ID'] = data_df['ID']

        data_groupby = data_df.groupby('ID', sort=False)
        labels_groupby = labels_df.groupby('ID', sort=False)

        val_indices = np.random.choice(len(data_groupby), size = int(0.2 * len(data_groupby)))

        tr_data_eng_arr = []
        tr_labels_eng_arr = []

        val_data_eng_arr = []
        val_labels_eng_arr = []

        for i in range(len(data_groupby)):
            if i in val_indices:
                val_data_eng_arr.append(data_groupby.get_group(i+1))
            else:
                tr_data_eng_arr.append(data_groupby.get_group(i+1))

        for i in range(len(labels_groupby)):
            if i in val_indices:
                val_labels_eng_arr.append(labels_groupby.get_group(i+1))
            else:
                tr_labels_eng_arr.append(labels_groupby.get_group(i+1))

        tr_data_windows = []
        tr_label_windows = []
        for index in range(len(tr_data_eng_arr)):
            tr_data_arr = tr_data_eng_arr[index].to_numpy()
            tr_labels_arr = tr_labels_eng_arr[index].to_numpy()
            for t in range(tr_data_arr.shape[0] - window_length + 1):
                tr_data_windows.append(tr_data_arr[t:t+window_length, :])
                tr_label_windows.append(tr_labels_arr[t+window_length - 1, 0])

        val_data_windows = []
        val_label_windows = []
        for index in range(len(val_data_eng_arr)):
            val_data_arr = val_data_eng_arr[index].to_numpy()
            val_labels_arr = val_labels_eng_arr[index].to_numpy()
            for t in range(val_data_arr.shape[0] - window_length + 1):
                val_data_windows.append(val_data_arr[t:t+window_length, :])
                val_label_windows.append(val_labels_arr[t+window_length - 1, 0])

        return np.array(tr_data_windows), np.array(tr_label_windows), np.array(val_data_windows), np.array(val_label_windows)

    else:

        labels_df['ID'] = data_df['ID']

        data_groupby = data_df.groupby('ID', sort=False)
        labels_groupby = labels_df.groupby('ID', sort=False)
        data_eng_arr = []
        labels_eng_arr = []

        for i in range(len(data_groupby)):
            data_eng_arr.append(data_groupby.get_group(i+1))

        for i in range(len(labels_groupby)):
            labels_eng_arr.append(labels_groupby.get_group(i+1))

        data_windows = []
        label_windows = []
        for index in range(len(data_eng_arr)):
            data_arr = data_eng_arr[index].to_numpy()
            labels_arr = labels_eng_arr[index].to_numpy()
            data_windows.append(data_arr[-window_length:, :])
            label_windows.append(labels_arr[-1, 0])

        return np.array(data_windows), np.array(label_windows)

# PLOT TRAIN AND VALIDATION LOSS
def plot_loss(fit_history):
    plt.figure(figsize=(13,5))
    plt.plot(range(1, len(fit_history.history['loss'])+1), fit_history.history['loss'], label='Training Loss')
    plt.plot(range(1, len(fit_history.history['val_loss'])+1), fit_history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# TESTING FUNCTION
def test(actual, pred, mode = 'Test'):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    variance = r2_score(actual, pred)
    print(mode + ' set RMSE: ' + str(rmse) + ', R2: ' + str(variance))

window_length = 20
cnn_tr_data, cnn_tr_labels, cnn_val_data, cnn_val_labels = get_windows(train_full_df, train_labels_full_df, window_length, mode='train')
cnn_test_data, cnn_test_labels = get_windows(test_df, test_labels_df, 20, mode = 'test')

cnn_tr_labels = np.expand_dims(cnn_tr_labels, axis=1)
cnn_val_labels = np.expand_dims(cnn_val_labels, axis=1)
cnn_test_labels = np.expand_dims(cnn_test_labels, axis=1)

# cnn = Sequential()
# cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, cnn_tr_data.shape[2])))
# cnn.add(MaxPool1D(pool_size = 2, padding = 'same', strides = 2))
# cnn.add(Convolution1D(64, 3, activation='relu'))
# cnn.add(MaxPool1D(pool_size = 2, padding = 'same', strides = 2))
# cnn.add(Convolution1D(32, 3, activation='relu'))
# cnn.add(MaxPool1D(pool_size = 2, padding = 'same', strides = 2))
# cnn.add(LSTM(128, activation = 'relu', return_sequences = True))
# cnn.add(LSTM(64, activation = 'relu', return_sequences = True))
# cnn.add(LSTM(32, activation = 'relu'))
# # cnn.add(GlobalAveragePooling1D(data_format = 'channels_last', keepdims = False))
# cnn.add(Dense(1))

cnn = Sequential()
cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, cnn_tr_data.shape[2])))
cnn.add(Dropout(0.55))
cnn.add(Convolution1D(64, 3, activation='relu'))
cnn.add(Dropout(0.55))
cnn.add(Convolution1D(22, 3, activation='relu'))
cnn.add(Dropout(0.55))
cnn.add(LSTM(128, activation = 'relu', return_sequences = True))
cnn.add(Dropout(0.55))
cnn.add(LSTM(64, activation = 'relu', return_sequences = True))
cnn.add(Dropout(0.55))
cnn.add(LSTM(22, activation = 'relu', return_sequences = True))
cnn.add(Dropout(0.55))
cnn.add(Bidirectional(LSTM(256, activation = 'relu', return_sequences = True)))
cnn.add(Dropout(0.55))
cnn.add(Bidirectional(LSTM(512, activation = 'relu')))
cnn.add(Dense(1))


# lstm = Sequential()
# lstm.add(LSTM(128, activation = 'relu', return_sequences = True, input_shape=(window_length, cnn_tr_data.shape[2])))
# lstm.add(LSTM(64, activation = 'relu', return_sequences = True))
# lstm.add(LSTM(32, activation = 'relu'))
# lstm.add(Dense(1))

cnn.compile(loss='mean_squared_error', optimizer = 'adam')   
cnn_history = cnn.fit(cnn_tr_data, cnn_tr_labels, epochs=75, validation_data = (cnn_val_data, cnn_val_labels), batch_size = 256)
# plot_loss(cnn_history)

train_cnn_pred = cnn.predict(cnn_tr_data)
test(cnn_tr_labels, train_cnn_pred, 'Train')

test_cnn_pred = cnn.predict(cnn_test_data)
test(cnn_test_labels, test_cnn_pred)

cnn.save_weights('cnn_weights.h5')

# lstm.compile(loss='mean_squared_error', optimizer='adam')  
# lstm_history = lstm.fit(cnn_tr_data, cnn_tr_labels, epochs=25, validation_data = (cnn_val_data, cnn_val_labels), batch_size = 128)
# # plot_loss(lstm_history)

# train_cnn_pred = lstm.predict(cnn_tr_data)
# test(cnn_tr_labels, train_cnn_pred, 'Train')

# test_cnn_pred = lstm.predict(cnn_test_data)
# test(cnn_test_labels, test_cnn_pred)