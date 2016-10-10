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
#train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.15/0.85, random_state=42)

# Data preprocessing
featurewise_mean = np.mean(train_data, axis=0)
featurewise_std = np.std(train_data, axis=0)
train_data = (train_data - featurewise_mean)/featurewise_std
test_data = (test_data - featurewise_mean)/featurewise_std
#val_data = (val_data - featurewise_mean)/featurewise_std

# Build neural network
input = tflearn.input_data(shape=[None, 8])

dense1 = tflearn.fully_connected(input, 50, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)

dense2 = tflearn.fully_connected(dense1, 20, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)

sum = tflearn.fully_connected(dense2, 1, activation='linear',
								bias=True, weights_init='truncated_normal', bias_init='zeros',
								regularizer='L2', weight_decay=0.001)

net = tflearn.regression(sum, optimizer='adam', 
						loss='mean_square', metric=None, 
						learning_rate=0.02)

# Define model
model = tflearn.DNN(net, 
					tensorboard_verbose=0, tensorboard_dir='summaries', 
					checkpoint_path='checkpoints/final-model/checkpoints')

# Start training (apply gradient descent algorithm)
model.fit(train_data, train_labels, 
			n_epoch=200, batch_size=None, validation_set=(test_data, test_labels), 
			show_metric=False, snapshot_step=500, run_id="final-model_2layers-50-20_relu_adam_lr-0.02")

# Test
print model.evaluate(test_data, test_labels)

# Plot a graph of testing data labels vs predictions
pred_labels = model.predict(test_data)

import matplotlib 
import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(test_labels[:20], pred_labels[:20])
plt.xlabel("test")
plt.ylabel("pred")
plt.show()