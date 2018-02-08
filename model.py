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
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
# Data preprocessing: normalization
model.add(Lambda(lambda x:(x / 255.0) - 0.5, input_shape=(160,320,3)))
# Data preprocessing: Cropping
model.add(Cropping2D(cropping=((70,25),(0,0))))
# Nvida Model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(Convolution2D(64,5,5,activation="relu"))
model.add(Convolution2D(64,5,5,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
exit()