from http import client
from pyexpat import model
from re import X
from turtle import window_width
from unicodedata import name
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Convolution1D, MaxPool1D, GlobalAveragePooling1D, Dense, Dropout, Bidirectional, Flatten
from keras import Sequential, backend
from keras.layers import LSTM, Dense, Dropout, Lambda, Input, Permute, RepeatVector, multiply, Concatenate, Reshape, Attention
from keras.models import Model

from sklearn.preprocessing import MinMaxScaler

train_data_full_df = pd.read_csv('../data_analysis/fd004/fd004-scaled_train.csv', sep=' ')
test_data_df = pd.read_csv('../data_analysis/fd004/fd004-scaled_test.csv', sep=' ')

train_labels_full_df = pd.read_csv('../data_analysis/fd004/fd004-training_labels.csv', sep=' ')
test_labels_df = pd.read_csv('../data_analysis/fd004/fd004-testing_labels.csv', sep=' ')
test_labels_at_break_df = pd.read_csv('../TED/CMAPSSData/RUL_FD004.txt', sep=' ', header=None)

# print(train_data_full_df.shape)
# print(test_data_df.shape)
# print(train_labels_full_df.shape)
# print(test_labels_df.shape)
# print(test_labels_at_break_df.shape)

train_full_df = train_data_full_df.copy()
test_df = test_data_df.copy()
train_labels_full_df = train_labels_full_df.copy().clip(upper=125)
test_labels_df = test_labels_df.copy()

used_sensors = train_full_df.columns
ms_used = used_sensors[2:]

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

val_indices = np.random.choice(len(train_groupby_full_df), size = int(0.25 * len(train_groupby_full_df)))

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

        val_indices = np.random.choice(len(data_groupby), size = int(0.1 * len(data_groupby)))

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
            if data_arr.shape[0] - window_length + 1 > 0:
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
    # plt.savefig("./loss_history-trend_lstm_01.png")
    plt.show()

# TESTING FUNCTION
def testing(actual, pred, mode = 'Test'):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    variance = r2_score(actual, pred)
    print(mode + ' set RMSE: ' + str(rmse) + ', R2: ' + str(variance))

window_length = 30
X_train, y_train, X_val, y_val = get_windows(train_full_df, train_labels_full_df, window_length, mode='train')
X_test, y_test = get_windows(test_df, test_labels_df, window_length, mode = 'test')

y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)
y_test = np.expand_dims(y_test, axis=1)

X_train = X_train[:,:,2:]
X_val = X_val[:,:,2:]
X_test = X_test[:,:,2:]


################### MLP ###############################################################  

# def mean_and_polynomial_fitting(data_set):
#     mean_trends = []
#     for window in data_set:
#         sensor_means = np.mean(window, axis = 0)
#         coefs = []
#         for i in range(window.shape[-1]):
#             coefs.append(np.polyfit(range(window.shape[0]), window[:,i], 1)[0])
#         mean_trend = np.concatenate((sensor_means, np.array(coefs)), axis = 0)
#         mean_trends.append(mean_trend)
#     return np.array(mean_trends)

# X_train = mean_and_polynomial_fitting(X_train)
# X_val = mean_and_polynomial_fitting(X_val)
# X_test = mean_and_polynomial_fitting(X_test)

# scale = preprocessing.MinMaxScaler()
# X_train = scale.fit_transform(X_train)
# X_val = scale.fit_transform(X_val)
# X_test = scale.fit_transform(X_test)

# mlp_model = Sequential()
# mlp_model.add(Dense(64, activation = 'relu', input_dim = X_train.shape[-1]))
# mlp_model.add(Dropout(0.25))
# mlp_model.add(Dense(128, activation = 'relu'))
# mlp_model.add(Dropout(0.25))
# mlp_model.add(Dense(256, activation = 'relu'))
# mlp_model.add(Dense(1))

# mlp_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0005))   
# mlp_history = mlp_model.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(mlp_history)

# train_mlp_pred = mlp_model.predict(X_train)
# testing(y_train, train_mlp_pred, 'Train')

# test_mlp_pred = mlp_model.predict(X_test)
# testing(y_test, test_mlp_pred)

# mlp_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00025), loss = 'mean_squared_error')
# mlp_history = mlp_model.fit(train_set_df[ms_used].values, train_set_labels, 
#                             validation_data = (val_set_df[ms_used].values, val_labels), 
#                             epochs = 30, batch_size = 256)                       
# plot_loss(mlp_history)
# train_mlp_pred = mlp_model.predict(train_data_full_df[ms_used].values)
# testing(train_labels_full, train_mlp_pred, 'Train')

# test_mlp_pred = mlp_model.predict(test_at_break_df[ms_used].values)
# testing(test_labels_at_break, test_mlp_pred)



################### CNN ###############################################################

# cnn = Sequential()
# cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, X_train.shape[2])))
# cnn.add(Convolution1D(64, 3, activation='relu'))
# cnn.add(Convolution1D(22, 3, activation='relu'))
# cnn.add(GlobalAveragePooling1D(data_format = 'channels_last', keepdims = False))
# cnn.add(Dense(64, activation = 'relu'))
# cnn.add(Dense(128, activation = 'relu'))
# cnn.add(Dense(1))

# cnn.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.001))   
# cnn_history = cnn.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(cnn_history)

# train_cnn_pred = cnn.predict(X_train)
# testing(y_train, train_cnn_pred, 'Train')

# test_cnn_pred = cnn.predict(X_test)
# testing(y_test, test_cnn_pred)


############## LSTM ##############################################################

# lstm = Sequential()
# lstm.add(LSTM(128, activation = 'tanh', return_sequences = True, input_shape=(window_length, X_train.shape[2])))
# lstm.add(LSTM(64, activation = 'tanh', return_sequences = True))
# lstm.add(LSTM(32, activation = 'tanh'))
# lstm.add(Dense(1))

# lstm.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.001))  
# lstm_history = lstm.fit(X_train, y_train, epochs=100, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(lstm_history)

# train_cnn_pred = lstm.predict(X_train)
# testing(y_train, train_cnn_pred, 'Train')

# test_cnn_pred = lstm.predict(X_test)
# testing(y_test, test_cnn_pred)

############## CNN + LSTM (Sequential) ########################################################

# cnn = Sequential()
# cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, X_train.shape[2])))
# cnn.add(Dropout(0.55))
# cnn.add(Convolution1D(64, 3, activation='relu'))
# cnn.add(Dropout(0.55))
# cnn.add(Convolution1D(22, 3, activation='relu'))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(128, activation = 'tanh', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(64, activation = 'tanh', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(22, activation = 'tanh', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(Bidirectional(LSTM(256, activation = 'tanh', return_sequences = True)))
# cnn.add(Dropout(0.55))
# cnn.add(Bidirectional(LSTM(512, activation = 'tanh')))
# cnn.add(Dense(1))

# cnn = Sequential()
# cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, X_train.shape[2])))
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

# cnn = Sequential()
# cnn.add(LSTM(
#          input_shape=(window_length, X_train.shape[-1]),
#          units=100,
#          return_sequences=True))
# cnn.add(Dropout(0.2))

# cnn.add(Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# cnn.add(MaxPool1D(pool_size=2))

# cnn.add(LSTM(
#           units=50,
#           return_sequences=False))
# cnn.add(Dropout(0.2))
# cnn.add(Dense(1))
# cnn.compile(loss='mean_squared_error', optimizer='adam')

# cnn.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.001))   
# cnn_history = cnn.fit(X_train, y_train, epochs=100, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(cnn_history)

# train_cnn_pred = cnn.predict(X_train)
# testing(y_train, train_cnn_pred, 'Train')

# test_cnn_pred = cnn.predict(X_test)
# testing(y_test, test_cnn_pred)

############ LSTM + CNN (Concatenation) #######################################################

# def cnn_lstm():
#     input_data = Input(shape=(window_length, X_train.shape[-1]))
#     cnn_layer1 = Convolution1D(64, kernel_size = 3)(input_data)
#     cnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer1)
#     cnn_layer3 = Convolution1D(32, kernel_size = 3)(cnn_layer2)
#     cnn_layer4 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer3)
#     cnn_layer5 = Flatten()(cnn_layer4)
#     cnn_layer6 = Dense(10, activation = 'relu')(cnn_layer5)
#     lstm_layer1 = LSTM(128, activation = 'tanh', return_sequences = True)(input_data)
#     lstm_layer2 = LSTM(64, activation = 'tanh', return_sequences = True)(lstm_layer1)
#     lstm_layer3 = LSTM(32, activation = 'tanh')(lstm_layer2)
#     lstm_layer4 = Dense(10, activation = 'relu')(lstm_layer3)
#     merged = Concatenate(axis = 1)([cnn_layer6, lstm_layer4])
#     ffnn_layer1 = Dense(128, activation = 'relu', kernel_regularizer='l2')(merged)
#     ffnn_layer2 = Dropout(0.2)(ffnn_layer1)
#     ffnn_layer3 = Dense(32, activation = 'relu', kernel_regularizer='l2')(ffnn_layer2)
#     out = Dense(1)(ffnn_layer3)
#     return Model(input_data, out)

# client_model = cnn_lstm()
# client_model.summary()
# client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0005))

# history = client_model.fit(X_train, y_train, epochs=30, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(history)

# train_pred = client_model.predict(X_train)
# testing(y_train, train_pred, 'Train')

# test_pred = client_model.predict(X_test) 
# testing(y_test, test_pred)

########################## MEAN_TRENDS + LSTM #######################################

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

train_extracted = mean_and_polynomial_fitting(X_train)
val_extracted = mean_and_polynomial_fitting(X_val)
test_extracted = mean_and_polynomial_fitting(X_test)

train_extracted = np.array(train_extracted)
val_extracted = np.array(val_extracted)
test_extracted = np.array(test_extracted)

scale = preprocessing.MinMaxScaler()
train_extracted = scale.fit_transform(train_extracted)
val_extracted = scale.fit_transform(val_extracted)
test_extracted = scale.fit_transform(test_extracted)

def trend_lstm():
    input_data = Input(shape=(window_length, X_train.shape[-1]))
    input_extracted = Input(shape=(train_extracted.shape[-1],))
    mlp_layer1 = Dense(64, activation = 'relu')(input_extracted)
    mlp_layer2 = Dropout(0.25)(mlp_layer1)
    mlp_layer3 = Dense(10, activation = 'relu')(mlp_layer2)
    lstm_layer1 = LSTM(64, activation = 'tanh', return_sequences = True)(input_data)
    lstm_layer2 = LSTM(32, activation = 'tanh')(lstm_layer1)
    lstm_layer3 = Dense(10, activation = 'relu')(lstm_layer2)
    merged = Concatenate(axis = 1)([mlp_layer3, lstm_layer3])
    ffnn_layer1 = Dense(32, activation = 'relu', kernel_regularizer='l2')(merged)
    ffnn_layer2 = Dropout(0.25)(ffnn_layer1)
    out = Dense(1)(ffnn_layer2)
    return Model([input_data, input_extracted], out)

client_model = trend_lstm()
client_model.summary()
client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0005))

# callback = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     min_delta=0,
#     patience=3,
#     mode='min',
#     baseline=None,
#     restore_best_weights=True
# )

history = client_model.fit([X_train, train_extracted], y_train, epochs=30, validation_data = ([X_val, val_extracted], y_val), batch_size = 64)
plot_loss(history)

train_pred = client_model.predict([X_train, train_extracted])
testing(y_train, train_pred, 'Train')

test_pred = client_model.predict([X_test, test_extracted]) 
testing(y_test, test_pred)

########################## MEAN_TRENDS + CNN #######################################

# def mean_and_polynomial_fitting(data_set):
#     mean_trends = []
#     for window in data_set:
#         sensor_means = np.mean(window, axis = 0)
#         coefs = []
#         for i in range(window.shape[-1]):
#             coefs.append(np.polyfit(range(window.shape[0]), window[:,i], 1)[0])
#         mean_trend = np.concatenate((sensor_means, np.array(coefs)), axis = 0)
#         mean_trends.append(mean_trend)
#     return np.array(mean_trends)

# train_extracted = mean_and_polynomial_fitting(X_train)
# val_extracted = mean_and_polynomial_fitting(X_val)
# test_extracted = mean_and_polynomial_fitting(X_test)

# scale = preprocessing.MinMaxScaler()
# train_extracted = scale.fit_transform(train_extracted)
# val_extracted = scale.fit_transform(val_extracted)
# test_extracted = scale.fit_transform(test_extracted)

# def trend_cnn():
#     input_data = Input(shape=(window_length, X_train.shape[-1]))
#     input_extracted = Input(shape=(train_extracted.shape[-1],))
#     cnn_layer1 = Convolution1D(64, kernel_size = 3)(input_data)
#     cnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer1)
#     cnn_layer3 = Convolution1D(32, kernel_size = 3)(cnn_layer2)
#     cnn_layer4 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer3)
#     cnn_layer5 = Flatten()(cnn_layer4)
#     cnn_layer6 = Dense(10, activation = 'relu')(cnn_layer5)
#     linear_layer1 = Dense(64, activation = 'relu')(input_extracted)
#     linear_layer2 = Dropout(0.25)(linear_layer1)
#     linear_layer3 = Dense(10, activation = 'relu')(linear_layer2)
#     linear_layer4 = Dropout(0.25)(linear_layer3)
#     merged = Concatenate(axis = 1)([cnn_layer6, linear_layer4])
#     ffnn_layer1 = Dense(128, activation = 'relu')(merged)
#     ffnn_layer2 = Dropout(0.25)(ffnn_layer1)
#     ffnn_layer3 = Dense(32, activation = 'relu')(ffnn_layer2)
#     out = Dense(1)(ffnn_layer3)
#     return Model([input_data, input_extracted], out)

# client_model = trend_cnn()
# client_model.summary()
# client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0005))

# history = client_model.fit([X_train, train_extracted], y_train, epochs=30, validation_data = ([X_val, val_extracted], y_val), batch_size = 256)
# plot_loss(history)

# train_pred = client_model.predict([X_train, train_extracted])
# testing(y_train, train_pred, 'Train')

# test_pred = client_model.predict([X_test, test_extracted]) 
# testing(y_test, test_pred)