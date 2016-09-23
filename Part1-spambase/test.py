import numpy as np
import tflearn

from tflearn.data_utils import load_csv
data, labels = load_csv('spambase.data', categorical_labels=True, n_classes=2)

def preprocess(data):
	return

data = preprocess(data)

# Build neural network
net = tflearn.input_data(shape=[None, 57])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)