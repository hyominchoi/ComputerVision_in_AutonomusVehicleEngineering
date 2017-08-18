import cv2
import csv
import numpy as np
import os
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('/home/carnd/Udacity_Self_Driving_Car/CarND-Behavioral-Cloning-P3/data/data_original/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]
with open('/home/carnd/Udacity_Self_Driving_Car/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

current_path = '/home/carnd/Udacity_Self_Driving_Car/CarND-Behavioral-Cloning-P3/data/IMG/'

def generator(samples, batch_size=32):
    delta = 0.1
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name =  current_path + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                flip_image = np.fliplr(center_image)
                images.append(flip_image)
                angles.append(-center_angle)
                
                name_left = current_path + batch_sample[1].split('/')[-1]
                """
                left_image = cv2.imread(name_left)
                left_angle = center_angle + delta
                images.append(left_image)
                angles.append(left_angle)
                
                right_image = cv2.imread(current_path + batch_sample[2].split('/')[-1])
                right_angle = center_angle - delta
                images.append(right_image)
                angles.append(right_angle)
                """
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((65, 20), (40,40)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255 - .5))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch = int(len(train_samples) * 2), 
                    validation_data = validation_generator, 
                    nb_val_samples = int(len(validation_samples) * 2), 
                    nb_epoch = 5)

model.save('model.h5')
