import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

def loadImage(source_path):
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	image =cv2.imread(current_path)
	return image

lines = []
with open('./data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	for line in reader:
		lines.append(line)
lines.pop(0)
images = []
measurements = []
for line in lines:
	measurement = float(line[3])
	left_correction = 0.24
	right_correction = 0.27
	#  Center Images
	source_path = line[0]
	center_image = loadImage(source_path)
	images.append(center_image)
	measurements.append(measurement)

	# flipped Images
	# images.append(cv2.flip(center_image,1))
	# measurements.append(-measurement)

	#  left Images
	source_path = line[1]
	images.append(loadImage(source_path))
	measurements.append(measurement + left_correction)
	
	#  right Images
	source_path = line[2]
	images.append(loadImage(source_path))
	measurements.append(measurement - right_correction)


X_train = np.array(images,dtype = np.float32)
y_train = np.array(measurements)
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
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
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(200))
# model.add(Dropout(0.5))
model.add(Dense(100))
# model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
exit()