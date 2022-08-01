import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Convolution1D, GlobalAveragePooling1D, Dense, Dropout
import keras_tuner as kt

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV

# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

train_data_df = pd.read_csv('../data_analysis/fd001/fd001-raw_train.csv', sep=' ')
test_data_df = pd.read_csv('../data_analysis/fd001/fd001-raw_test.csv', sep=' ')

train_labels_df = pd.read_csv('../data_analysis/fd001/fd001-training_labels.csv', sep=' ')
test_labels_df = pd.read_csv('../data_analysis/fd001/fd001-testing_labels.csv', sep=' ')
test_labels_at_break_df = pd.DataFrame(pd.read_csv('../TED/CMAPSSData/RUL_FD001.txt', sep=' ', header=None)[0])
test_labels_at_break_df.columns = ['RUL']

test_at_break_df = test_data_df.groupby(['ID'], sort=False).last().reset_index()
train_labels_df = train_labels_df.clip(upper = 125)
test_labels_df = test_labels_df.clip(upper = 125)

train_labels_df['ID'] = train_data_df['ID']
test_labels_df['ID'] = test_data_df['ID']

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
                tr_label_windows.append(tr_labels_arr[t+window_length - 1])

        val_data_windows = []
        val_label_windows = []
        for index in range(len(val_data_eng_arr)):
            val_data_arr = val_data_eng_arr[index].to_numpy()
            val_labels_arr = val_labels_eng_arr[index].to_numpy()
            for t in range(val_data_arr.shape[0] - window_length + 1):
                val_data_windows.append(val_data_arr[t:t+window_length, :])
                val_label_windows.append(val_labels_arr[t+window_length - 1])

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

window_length = 20
mlp_tr_data, mlp_tr_labels, mlp_val_data, mlp_val_labels = get_windows(train_data_df, train_labels_df, window_length, mode='train')
mlp_test_data, mlp_test_labels = get_windows(test_data_df, test_labels_df, 20, mode = 'test')

mlp_tr_data = mlp_tr_data.reshape(mlp_tr_data.shape[0], -1)
mlp_val_data = mlp_val_data.reshape(mlp_val_data.shape[0], -1)
mlp_test_data = mlp_test_data.reshape(mlp_test_data.shape[0], -1)

mlp_tr_labels = np.expand_dims(mlp_tr_labels, axis=1)
mlp_val_labels = np.expand_dims(mlp_val_labels, axis=1)
mlp_test_labels = np.expand_dims(mlp_test_labels, axis=1)

ms_used = train_data_df.columns[2:]

print(mlp_test_data[:,2:].shape)
print(mlp_test_labels.shape)

# print(train_data_df.shape)
# print(train_labels_df.shape)
# print(test_data_df.shape)
# print(test_labels_df.shape)
# print(test_labels_at_break_df.shape)
# print(test_at_break_df.shape)

# train_groupby_df = train_data_df.groupby(['ID'], sort = False)
# train_labels_groupby_df = train_labels_df.groupby(['ID'], sort = False)
# val_indices = np.random.choice(len(train_groupby_df), size = int(0.2 * len(train_groupby_df)))

# val_arr = []
# val_lab = []
# tr_arr = []
# tr_lab = []
# for i in range(len(train_groupby_df)):
#     if i in val_indices:
#         val_arr.append(train_groupby_df.get_group(i+1))
#         val_lab.append(train_labels_groupby_df.get_group(i+1)['RUL'])
#     else:
#         tr_arr.append(train_groupby_df.get_group(i+1))
#         tr_lab.append(train_labels_groupby_df.get_group(i+1)['RUL'])

# tr_df = tr_arr[0]
# val_df = val_arr[0]
# tr_lab_df = tr_lab[0]
# val_lab_df = val_lab[0]

# for i in range(1, len(tr_arr)):
#     tr_df = pd.concat([tr_df, tr_arr[i]])
#     tr_lab_df = pd.concat([tr_lab_df, tr_lab[i]])
# for i in range(1, len(val_arr)):
#     val_df = pd.concat([val_df, val_arr[i]])
#     val_lab_df = pd.concat([val_lab_df, val_lab[i]])

# print(tr_df.shape)
# print(tr_lab_df.shape)
# print(val_df.shape)
# print(val_lab_df.shape)
# print(train_data_df.shape)
# print(train_labels_df.shape)
# print(test_labels_df.shape)
# print(test_labels_at_break_df.shape)

# def mlp_model_builder(hp):

#     hp_units1 = hp.Int('units1', min_value=32, max_value=128, step=32)
#     hp_units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
#     hp_units3 = hp.Int('units3', min_value=32, max_value=128, step=32)

#     hp_dropout = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4])
#     hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.005, 0.01, 0.05])

#     mlp_model = Sequential()
#     mlp_model.add(Dense(units = hp_units1, activation = 'relu', input_dim = train_set_df[ms_used].values.shape[1]))
#     mlp_model.add(Dropout(hp_dropout))
#     mlp_model.add(Dense(units = hp_units2, activation = 'relu'))
#     mlp_model.add(Dropout(hp_dropout))
#     mlp_model.add(Dense(units = hp_units3 , activation = 'relu'))
#     mlp_model.add(Dropout(hp_dropout))
#     mlp_model.add(Dense(1, activation = 'relu'))

#     mlp_model.compile(optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate),
#                 loss=keras.losses.MeanSquaredError())

#     return mlp_model

# mlp_tuner = kt.BayesianOptimization(mlp_model_builder,
#                                     objective='val_loss',
#                                     max_trials = 9,
#                                     directory='baseline_models',
#                                     project_name='mlp')
                            
# mlp_tuner.search(train_set_df[ms_used].values, train_set_labels.squeeze(), epochs=100, validation_data = (val_set_df[ms_used].values, val_labels.squeeze()), batch_size = 256)
# best_mlp_hps = mlp_tuner.get_best_hyperparameters(num_trials=1)[0]

# best_mlp_model = mlp_tuner.hypermodel.build(best_mlp_hps)
# mlp_history = best_mlp_model.fit(train_set_df[ms_used].values, train_set_labels.squeeze(), epochs=100, validation_data = (val_set_df[ms_used].values, val_labels.squeeze()), batch_size = 256)

mlp_model = Sequential()
mlp_model.add(Dense(16, activation = 'relu', input_dim = mlp_tr_data[:,2:].shape[-1]))
mlp_model.add(Dense(32, activation = 'relu'))
mlp_model.add(Dense(64, activation = 'relu'))
mlp_model.add(Dense(1))

mlp_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001), loss = keras.losses.MeanSquaredError())

mlp_history = mlp_model.fit(mlp_tr_data[:,2:], mlp_tr_labels, 
                            validation_data = (mlp_val_data[:,2:], mlp_val_labels), 
                            epochs = 150, batch_size = 128)


# PLOT TRAIN AND VALIDATION LOSS
def plot_loss(fit_history):
    plt.figure(figsize=(13,5))
    plt.plot(range(1, len(fit_history.history['loss'])+1), fit_history.history['loss'], label='train')
    plt.plot(range(1, len(fit_history.history['val_loss'])+1), fit_history.history['val_loss'], label='validate')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss(mlp_history)

# TESTING FUNCTION
def testing(actual, pred, mode = 'Test'):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    variance = r2_score(actual, pred)
    print(mode + ' set RMSE: ' + str(rmse) + ', R2: ' + str(variance))

train_full_pred = mlp_model.predict(mlp_tr_data[:,2:])
print(train_full_pred.shape)
testing(mlp_tr_labels.squeeze(), train_full_pred, 'Train')

test_at_break_pred = mlp_model.predict(mlp_test_data[:,2:])
testing(mlp_test_labels.squeeze(), test_at_break_pred)

