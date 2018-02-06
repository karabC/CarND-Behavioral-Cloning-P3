import csv
import cv2
import numpy as np
lines = []
with open('./data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	for line in reader:
		lines.append(line)
lines.pop(0)
images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image =cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images,dtype = np.float32)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#  Keras 1.2.1 Bug 
#  Monkey-patch Keras with the fix.
#  Reference: https://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error
from keras.wrappers.scikit_learn import BaseWrapper
import copy

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params


def create_model(dropout_rate=0.0):
	model = Sequential()
	# Data preprocessing: normalization
	model.add(Lambda(lambda x:(x / 255.0) - 0.5, input_shape=(160,320,3)))
	# Data preprocessing: Cropping
	model.add(Cropping2D(cropping=((70,25),(0,0))))
	# Nvida Model
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dropout(dropout_rate))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	model.compile(loss='mse', optimizer='adam')

# fit_params = dict(, nb_epoch=5)
model = KerasClassifier(build_fn=create_model ,nb_epoch=5, batch_size=10, verbose=0,validation_split=0.2, shuffle=True)

dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

model.save('model.h5')
exit()