import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
import sklearn
import os
import csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#read the .csv file to get de samples storing path,put them into a list "samples"
samples = []
with open('./mytrainingdata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
#split list(indicate the data) into training data and validation data.the portition is 20%
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# define the generator function to generate a batch training samples
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)#shuffle the samples
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            #according to the sample path to load the training data
            for batch_sample in batch_samples:
                center_image = cv2.imread('./mytrainingdata/IMG/' + batch_sample[0].split('/')[-1])
                left_image = cv2.imread('./mytrainingdata/IMG/' + batch_sample[1].split('/')[-1])
                right_image = cv2.imread('./mytrainingdata/IMG/' + batch_sample[2].split('/')[-1])
                center_angle = float(batch_sample[3])
                correction = 0.45
                left_angle = float(batch_sample[3]) + correction
                right_angle = float(batch_sample[3]) - correction
                #add center camera data into training dataset
                images.append(center_image)
                angles.append(center_angle)
                image_flipped = np.fliplr(center_image)
                images.append(image_flipped)
                angles.append(-center_angle)
                #add left camera data into training dataset
                images.append(left_image)
                angles.append(left_angle) 
                image_flipped = np.fliplr(left_image)
                images.append(image_flipped)
                angles.append(-left_angle)
                #add right camera data into training dataset
                images.append(right_image)
                angles.append(right_angle)
                image_flipped = np.fliplr(right_image)
                images.append(image_flipped)
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# create train generator function and validation generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# raw image size
row, col, ch = 160, 320, 3

# define the model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x / 255 - 0.5,input_shape=(row, col, ch), output_shape=(row, col, ch)))
#trim the input image to get more proper region
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Convolution2D(16, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.15))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))

model.add(Dense(1))

#choose the adam optimizer to optimize the mean square error
model.compile(loss='mse', optimizer='adam')
#use generators which are defined at the above code to generate training and vlidation data
#the epoch is set to be 5
history_object = model.fit_generator(train_generator, samples_per_epoch=len(
    train_samples) * 6, validation_data=validation_generator, nb_val_samples=len(validation_samples) * 6, nb_epoch=5, verbose=1)

#save the trained model
model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
