import numpy as np
import tflearn

# Load data
from tflearn.data_utils import load_csv
data, labels = load_csv('spambase.data', has_header=False, categorical_labels=True, n_classes=2)

# Shuffle and split data
data = np.array(data).astype(np.float)
labels = np.array(labels)

from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.15, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.15/0.85, random_state=42)

# Data preprocessing
featurewise_mean = np.mean(train_data, axis=0)
featurewise_std = np.std(train_data, axis=0)
train_data = (train_data - featurewise_mean)/featurewise_std
test_data = (test_data - featurewise_mean)/featurewise_std
val_data = (val_data - featurewise_mean)/featurewise_std

# Build neural network
input = tflearn.input_data(shape=[None, 57])

dense1 = tflearn.fully_connected(input, 50, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, keep_prob=0.5)

dense2 = tflearn.fully_connected(dropout1, 30, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, keep_prob=0.5)

softmax = tflearn.fully_connected(dropout2, 2, activation='softmax',
								bias=True, weights_init='truncated_normal', bias_init='zeros',
								regularizer='L2', weight_decay=0.001)

net = tflearn.regression(softmax, optimizer='adam', 
						loss='categorical_crossentropy', metric='accuracy', 
						learning_rate=0.002)

# Define model
model = tflearn.DNN(net, 
					tensorboard_verbose=0, tensorboard_dir='summaries/train-val-test', 
					checkpoint_path='checkpoints/checkpoints', best_checkpoint_path='best_checkpoints/train-val-test/2layer-50-30_relu_adam_online_lr-0.002/best_checkpoint')

# Start training (apply gradient descent algorithm)
model.fit(train_data, train_labels, 
			n_epoch=100, batch_size=None, validation_set=(val_data, val_labels), 
			show_metric=True, snapshot_step=100, run_id="2layer-50-30_relu_adam_online_lr-0.002")