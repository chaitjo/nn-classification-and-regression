import numpy as np
import tflearn

from tflearn.data_utils import load_csv
data, labels = load_csv('spambase.data', has_header=False, categorical_labels=True, n_classes=2)
data = np.array(data).astype(np.float)

# Define data preprocessing
spam_prep = tflearn.data_preprocessing.DataPreprocessing()
spam_prep.add_featurewise_zero_center()
spam_prep.add_featurewise_stdnorm()

# Build neural network
input = tflearn.input_data(shape=[None, 57], data_preprocessing=spam_prep)

dense1 = tflearn.fully_connected(input, 32, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, keep_prob=0.5)

dense2 = tflearn.fully_connected(dropout1, 32, activation='relu', 
								bias=True, weights_init='truncated_normal', bias_init='zeros', 
								regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, keep_prob=0.5)

softmax = tflearn.fully_connected(dropout2, 2, activation='softmax',
								bias=True, weights_init='truncated_normal', bias_init='zeros',
								regularizer='L2', weight_decay=0.001)

net = tflearn.regression(softmax, optimizer='adam', 
						loss='categorical_crossentropy', metric='accuracy', 
						learning_rate=0.001, 
						to_one_hot=True, n_classes=2)

# Define model
model = tflearn.DNN(net, 
					tensorboard_verbose=0, tensorboard_dir='runs/summaries', 
					checkpoint_path='runs/checkpoints', best_checkpoint_path='runs/best_checkpoint', best_val_accuracy=0.8)

# Start training (apply gradient descent algorithm)
# model.fit(data, labels, 
# 			n_epoch=10, batch_size=16, validation_set=0.2, 
# 			show_metric=True, shuffle=True, 
# 			snapshot_epoch=True, snapshot_step=100)

model.fit(data, labels, 
			n_epoch=10, batch_size=16, 
			show_metric=True)

# Test with dummy data
test = [[0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278]]
print model.predict(test)
