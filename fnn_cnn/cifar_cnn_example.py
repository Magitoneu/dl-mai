from __future__ import division
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
import numpy as np
import sys
from keras import backend as K
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.io

# Check https://keras.io/preprocessing/image/#imagedatagenerator-class:
#   - flow_from_directory
#   - image augmentation
# https://stackoverflow.com/questions/53037510/can-flow-from-directory-get-train-and-validation-data-from-the-same-directory-in
# https://github.com/jfilter/split-folders

print('Using Keras version', keras.__version__)


path_images='/home/magi/mai/s2/dl/lab/datasets/101_ObjectCategories'
path_annotation="/home/magi/mai/s2/dl/lab/datasets/Annotations"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=20,
        brightness_range=(0.7,1.3),
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '../datasets/Images_split/train',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../datasets/Images_split/val',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical')

# resolution
img_rows, img_cols, channels = 64, 64, 3

if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, channels)

# Define the NN architecture

# Two hidden layers
nn = Sequential()
nn.add(Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='SAME', activation='relu', input_shape=input_shape))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dense(128, activation='relu'))
nn.add(Dense(67, activation='softmax'))

# Compile the NN
# opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.0)
opt = optimizers.Adam(lr=0.001)
nn.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
nn.build(input_shape)
print(nn.summary())

# Start training
history = nn.fit_generator(
          train_generator,
          steps_per_epoch=13714/64,
          epochs=500,
          validation_data=validation_generator,
          validation_steps=1906/64)

# Evaluate the model with test set
# score = nn.evaluate(x_test, y_test, verbose=0)
# print('test loss:', score[0])
# print('test accuracy:', score[1])

weights_file = "weights-INDOOR_smallNN_" + ".hdf5"
nn.save_weights(weights_file, overwrite=True)

# Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('cifar10_cnn_accuracy.pdf')
plt.close()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('cifar10_cnn_loss.pdf')

sys.exit()
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Compute probabilities
Y_pred = nn.predict(x_test)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
# Plot statistics
print('Analysis of results')
target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

# Saving model and weights
from keras.models import model_from_json

nn_json = nn.to_json()
with open('nn.json', 'w') as json_file:
    json_file.write(nn_json)
weights_file = "weights-CIFAR10_" + str(score[1]) + ".hdf5"
nn.save_weights(weights_file, overwrite=True)

# Loading model and weights
json_file = open('nn.json', 'r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)
