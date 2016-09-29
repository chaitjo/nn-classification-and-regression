import numpy as np
import tflearn

from tflearn.data_utils import load_csv
data, labels = load_csv('spambase.data', has_header=False, categorical_labels=True, n_classes=2)

# Define data preprocessing
spam_prep = tflearn.data_preprocessing.DataPreprocessing()
spam_prep.add_featurewise_zero_center()
spam_prep.add_featurewise_stdnorm()

# Build neural network
net = tflearn.input_data(shape=[None, 57], data_preprocessing=spam_prep)
net = tflearn.fully_connected(net, 32, activation='linear', bias=True, weights_init='truncated_normal', bias_init='zeros', regularizer='L2', weight_decay=0.001)
net = tflearn.fully_connected(net, 32, activation='linear', bias=True, weights_init='truncated_normal', bias_init='zeros', regularizer='L2', weight_decay=0.001)
net = tflearn.dropout(net, keep_prob=0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

test = [0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278]
print model.predict(test)