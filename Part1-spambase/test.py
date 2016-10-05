import numpy as np
import tflearn
import sklearn

#Load data
from tflearn.data_utils import load_csv
data, labels = load_csv('spambase.data', has_header=False, categorical_labels=True, n_classes=2)

# Data preprocessing
data = np.array(data).astype(np.float)
labels = np.array(labels)
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(data)))
data_shuffled = data[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=42)

featurewise_mean = np.mean(train_data, axis=0)
featurewise_std = np.std(train_data, axis=0)

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

dense3 = tflearn.fully_connected(dropout2, 15, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)
dropout3 = tflearn.dropout(dense3, keep_prob=0.5)

softmax = tflearn.fully_connected(dropout3, 2, activation='softmax',
								bias=True, weights_init='truncated_normal', bias_init='zeros',
								regularizer='L2', weight_decay=0.001)

net = tflearn.regression(softmax, optimizer='adam', 
						loss='categorical_crossentropy', metric='accuracy', 
						learning_rate=0.001)

# Define model
model = tflearn.DNN(net, 
					tensorboard_verbose=0, tensorboard_dir='summaries', 
					checkpoint_path='checkpoints/checkpoints', best_checkpoint_path='best_checkpoints/best_checkpoint', best_val_accuracy=0.8)

# Start training (apply gradient descent algorithm)
model.fit(data, labels, 
			n_epoch=200, batch_size=None, validation_set=[val_data, val_labels], 
			show_metric=True, snapshot_step=100)

# Test with dummy data
test = [[0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278]]
print model.predict(test)
