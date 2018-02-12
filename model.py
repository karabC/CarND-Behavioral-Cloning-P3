import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def loadImage(source_path):
	filename = source_path.split('/')[-1]
	current_path = './data/IMG/' + filename
	imageBGR =cv2.imread(current_path)
	return imageBGR

# Define the angle correction
left_correction = 0.25
right_correction = 0.27

# Construct generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: 
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                measurement = batch_sample[3]
                name = batch_sample[0]
                center_image = loadImage(name)
                center_angle = float(measurement)
                images.append(center_image)
                angles.append(center_angle)

                images.append(cv2.flip(center_image,1))
                angles.append(-center_angle)

                left_image = loadImage(batch_sample[1])
                left_angle = float(measurement) + left_correction
                images.append(left_image)
                angles.append(left_angle)

                right_image = loadImage(batch_sample[2])
                right_angle = float(measurement) - right_correction
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Read the Data
lines = []
with open('./data/driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	for line in reader:
		lines.append(line)
lines.pop(0)
# compile and train the model using the generator function
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
# Data preprocessing: Cropping
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))
# Data preprocessing: normalization
model.add(Lambda(lambda x:(x / 255.0) - 0.5))
# Nvida Model
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
# model.add(Dense(200))
# model.add(Dropout(0.5))
model.add(Dense(100))
# model.add(Dropout(0.5))
model.add(Dense(50))
# model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= 
	             len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
exit()