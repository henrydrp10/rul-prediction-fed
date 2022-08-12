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

train_data_full_df = pd.read_csv('../data_analysis/fd001/fd001-scaled_train.csv', sep=' ')
test_data_df = pd.read_csv('../data_analysis/fd001/fd001-scaled_test.csv', sep=' ')

train_labels_full_df = pd.read_csv('../data_analysis/fd001/fd001-training_labels.csv', sep=' ')
test_labels_df = pd.read_csv('../data_analysis/fd001/fd001-testing_labels.csv', sep=' ')
test_labels_at_break_df = pd.read_csv('../TED/CMAPSSData/RUL_FD001.txt', sep=' ', header=None)

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
            if data_arr.shape[0] - window_length > 0:
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

################### MLP ###############################################################  

# mlp_model = Sequential()
# mlp_model.add(Dense(32, activation = 'relu', input_dim = train_data_full_df[ms_used].shape[-1]))
# mlp_model.add(Dropout(0.25))
# mlp_model.add(Dense(64, activation = 'relu'))
# mlp_model.add(Dropout(0.25))
# mlp_model.add(Dense(128, activation = 'relu'))
# mlp_model.add(Dense(1))

# mlp_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.003), loss = keras.losses.MeanSquaredError())

# mlp_history = mlp_model.fit(train_set_df[ms_used].values, train_set_labels, 
#                             validation_data = (val_set_df[ms_used].values, val_labels), 
#                             epochs = 75, batch_size = 128)
                        
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

# lstm.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.01))  
# lstm_history = lstm.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 512)
# plot_loss(lstm_history)

# train_cnn_pred = lstm.predict(X_train)
# testing(y_train, train_cnn_pred, 'Train')

# test_cnn_pred = lstm.predict(X_test)
# testing(y_test, test_cnn_pred)

############## CNN + LSTM ########################################################

# cnn = Sequential()
# cnn.add(Convolution1D(128, 3, activation='relu', input_shape = (window_length, X_train.shape[2])))
# cnn.add(Dropout(0.55))
# cnn.add(Convolution1D(64, 3, activation='relu'))
# cnn.add(Dropout(0.55))
# cnn.add(Convolution1D(22, 3, activation='relu'))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(128, activation = 'relu', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(64, activation = 'relu', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(LSTM(22, activation = 'relu', return_sequences = True))
# cnn.add(Dropout(0.55))
# cnn.add(Bidirectional(LSTM(256, activation = 'relu', return_sequences = True)))
# cnn.add(Dropout(0.55))
# cnn.add(Bidirectional(LSTM(512, activation = 'relu')))
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
# cnn_history = cnn.fit(X_train, y_train, epochs=50, validation_data = (X_val, y_val), batch_size = 256)
# plot_loss(cnn_history)

# train_cnn_pred = cnn.predict(X_train)
# testing(y_train, train_cnn_pred, 'Train')

# test_cnn_pred = cnn.predict(X_test)
# testing(y_test, test_cnn_pred)

############ LSTM + ATTENTION #######################################################

X_train = X_train[:,:,2:]
X_val = X_val[:,:,2:]
X_test = X_test[:,:,2:]

# sc = preprocessing.MinMaxScaler()

# for i in range(len(X_train)):
#     X_train[i] = sc.fit_transform(X_train[i])

# for i in range(len(X_val)):
#     X_val[i] = sc.fit_transform(X_val[i])

# for i in range(len(X_test)):
#     X_test[i] = sc.fit_transform(X_test[i])

regr = linear_model.LinearRegression() # feature of linear coefficient

def fea_extract(data): # feature extraction of two features
    fea = []
    # print(data.shape)
    x = np.array(range(data.shape[0]))
    for i in range(data.shape[1]):
        fea.append(np.mean(data[:,i]))
        # print(x.reshape(-1,1).shape)
        # print(np.ravel(data[:,i]).shape)
        regr.fit(x.reshape(-1,1),np.ravel(data[:,i]))
        # print(np.array(fea).shape)
        fea = fea+list(regr.coef_)
        # print(np.array(fea).shape)
    return fea

train_extracted = []
val_extracted = []
test_extracted = []

for window in X_train:
    train_extracted.append(fea_extract(window))

for window in X_val:
    val_extracted.append(fea_extract(window))

for window in X_test:
    test_extracted.append(fea_extract(window))

train_extracted = np.array(train_extracted)
val_extracted = np.array(val_extracted)
test_extracted = np.array(test_extracted)

# scale = preprocessing.MinMaxScaler()
# train_extracted = scale.fit_transform(train_extracted)
# val_extracted = scale.fit_transform(val_extracted)
# test_extracted = scale.fit_transform(test_extracted)

# train_extracted = X_train.copy()
# val_extracted = X_val.copy()
# test_extracted = X_test.copy()

# print(train_extracted.shape)
# print(test_extracted.shape)
# print(test_extracted.shape)


def temp_spatial_fusion():
    input_data = Input(shape=(window_length, X_train.shape[-1]))
    cnn_layer1 = Convolution1D(128, kernel_size = 3)(input_data)
    cnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer1)
    cnn_layer3 = Convolution1D(64, kernel_size = 3)(input_data)
    cnn_layer4 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer3)
    cnn_layer5 = Convolution1D(32, kernel_size = 3)(cnn_layer4)
    cnn_layer6 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer5)
    cnn_layer6 = Flatten()(cnn_layer6)
    cnn_layer6 = Dense(10, activation = 'relu')(cnn_layer6)
    attention = attention_3d_block(input_data)
    lstm_layer1 = LSTM(128, activation = 'tanh', return_sequences = True)(attention)
    lstm_layer2 = LSTM(64, activation = 'tanh', return_sequences = True)(lstm_layer1)
    lstm_layer3 = LSTM(32, activation = 'tanh', return_sequences = True)(lstm_layer2)
    lstm_layer3 = LSTM(32, activation = 'tanh')(lstm_layer2)
    lstm_layer3 = Dense(10, activation = 'relu')(lstm_layer3)
    merged = Concatenate(axis = 1)([cnn_layer6, lstm_layer3])
    # ffnn_layer1 = Convolution1D(256, kernel_size = 3)(merged)
    # ffnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(ffnn_layer1)
    # ffnn_layer3 = Flatten()(ffnn_layer2)
    ffnn_layer4 = Dense(128, activation = 'relu')(merged)
    ffnn_layer5 = Dropout(0.2)(ffnn_layer4)
    ffnn_layer6 = Dense(32, activation = 'relu')(ffnn_layer5)
    out = Dense(1)(ffnn_layer6)
    return Model(input_data, out)

# RMSE: 39.56266894113846
# SINGLE LSTM + ATTENTION MODEL: 
def single_attention_model():
    input_data = Input(shape=(window_length, X_train.shape[-1]))
    layer1 = attention_3d_block(input_data)
    layer2 = LSTM(100, activation = 'tanh', return_sequences = True)(layer1)
    layer3 = Dropout(0.5)(layer2)
    layer4 = Dense(30, activation = 'relu')(layer3)
    layer5 = Dense(20, activation = 'relu')(layer4)
    out = tf.squeeze(Dense(1)(layer5))
    return Model(input_data, out)

# CUSTOM ATTENTION BLOCK ----- https://github.com/ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, window_length))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(window_length, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    print(a_probs.shape)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

# RMSE: 18.612317489911245 (fd001)
# LSTM + ATTENTION ------- https://github.com/ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach
def model_attention():
    input_data = Input(shape=(window_length, X_train.shape[-1]))
    input_extracted = Input(shape = (train_extracted.shape[1],))
    ex_layer1 = Dense(50, activation = 'relu')(input_extracted)
    ex_layer2 = Dropout(0.2)(ex_layer1)
    ex_layer3 = Dense(10,activation = 'relu')(ex_layer2)
    att_layer1 = attention_3d_block(input_data)
    att_layer2 = LSTM(50, return_sequences=False)(att_layer1) 
    att_layer3 = Dense(50, activation='relu')(att_layer2)
    att_layer4 = Dropout(0.2)(att_layer3)
    att_layer5 = Dense(10, activation='relu')(att_layer4) 
    merged_models = Concatenate(axis = 1)([att_layer5, ex_layer3])
    merged_layer1 = Dropout(0.2)(merged_models) 
    merged_layer2 = Dense(1, activation='linear')(merged_layer1)
    model = Model([input_data, input_extracted], merged_layer2)
    return model


# RMSE: 37.14
# def combined_model():
#     input_data = Input(shape=(window_length, X_train.shape[-1]))
#     input_data2 = Input(shape=(window_length, X_train.shape[-1]))
#     input_extracted = Input(shape = (train_extracted.shape[1],))
#     ex_layer1 = Dense(50, activation = 'relu')(input_extracted)
#     ex_layer2 = Dropout(0.2)(ex_layer1)
#     ex_layer3 = Dense(10, activation = 'relu')(ex_layer2)
#     lstm_layer0 = attention_3d_block(input_data)
#     lstm_layer1 = LSTM(128, activation = 'tanh', return_sequences = True)(lstm_layer0)
#     lstm_layer2 = LSTM(64, activation = 'tanh', return_sequences = True)(lstm_layer1)
#     lstm_layer3 = LSTM(32, activation = 'tanh', return_sequences = False)(lstm_layer2)
#     lstm_layer4 = Dense(50, activation='relu')(lstm_layer3)
#     lstm_layer5 = Dropout(0.2)(lstm_layer4)
#     lstm_layer6 = Dense(10, activation='relu')(lstm_layer5)
#     cnn_layer1 = Convolution1D(128, kernel_size = 3)(input_data2)
#     cnn_layer2 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer1)
#     cnn_layer3 = Convolution1D(64, kernel_size = 3)(cnn_layer2)
#     cnn_layer4 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer3)
#     cnn_layer5 = Convolution1D(32, kernel_size = 3)(cnn_layer4)
#     cnn_layer6 = MaxPool1D(pool_size = 2, padding = 'same', strides = 2)(cnn_layer5)
#     cnn_layer7 = GlobalAveragePooling1D()(cnn_layer6)
#     ffnn_layer1 = Concatenate(axis = 1)([cnn_layer7, lstm_layer6, ex_layer3])
#     ffnn_layer2 = Dense(128, activation = 'relu')(ffnn_layer1)
#     ffnn_layer3 = Dense(32, activation = 'relu')(ffnn_layer2)
#     out = Dense(1)(ffnn_layer3)
#     return Model([input_data, input_data2, input_extracted], out)

client_model = temp_spatial_fusion()
client_model.summary()
client_model.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate = 0.0008))

# att_history = client_model.fit(X_train, y_train, epochs=15, validation_data = (X_val, y_val), batch_size = 100)
att_history = client_model.fit(X_train, y_train, epochs=15, validation_data = (X_val, y_val), batch_size = 64)
plot_loss(att_history)

train_pred = client_model.predict(X_train)
testing(y_train, train_pred, 'Train')

test_pred = client_model.predict(X_test) 
testing(y_test, test_pred)