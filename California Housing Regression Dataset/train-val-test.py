import numpy as np
import tflearn

# Load data
from tflearn.data_utils import load_csv
data, labels = load_csv('cal_housing.data', has_header=False)

# Shuffle and split data
data = np.array(data).astype(np.float)
labels = (np.array(labels).astype(np.float).reshape(len(data), 1))/1000

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.15, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.15/0.85, random_state=42)

# Data preprocessing
featurewise_mean = np.mean(train_data, axis=0)
featurewise_std = np.std(train_data, axis=0)
train_data = (train_data - featurewise_mean)/featurewise_std
test_data = (test_data - featurewise_mean)/featurewise_std
val_data = (val_data - featurewise_mean)/featurewise_std

# Build neural network
input = tflearn.input_data(shape=[None, 8])

dense1 = tflearn.fully_connected(input, 60, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)

dense2 = tflearn.fully_connected(dense1, 30, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)

dense3 = tflearn.fully_connected(dense2, 15, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)

sum = tflearn.fully_connected(dense3, 1, activation='linear',
								bias=True, weights_init='truncated_normal', bias_init='zeros',
								regularizer='L2', weight_decay=0.001)

net = tflearn.regression(sum, optimizer='adam', 
						loss='mean_square', metric=None, 
						learning_rate=0.05)

# Define model
model = tflearn.DNN(net, 
					tensorboard_verbose=0, tensorboard_dir='summaries/train-val-test', 
					checkpoint_path='checkpoints/train-val-test/3layers-60-30-15_relu_adam_lr-0.05/checkpoints')

# Start training (apply gradient descent algorithm)
model.fit(train_data, train_labels, 
			n_epoch=200, batch_size=None, validation_set=(val_data, val_labels), 
			show_metric=False, snapshot_step=500, run_id="3layers-60-30-15_relu_adam_lr-0.05")

# test = [[-122.230000,37.880000,41.000000,880.000000,129.000000,322.000000,126.000000,8.325200]]
# test = (test - featurewise_mean)/featurewise_std
# print model.predict(test)