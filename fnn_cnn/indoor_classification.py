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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from PIL import Image
import scipy.io

print('Using Keras version', keras.__version__)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #        shear_range=0.2,
    #        rotation_range=20,
    #        brightness_range=(0.7,1.3),
    #        horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# /home/nct01/nct01078
train_generator = train_datagen.flow_from_directory(
    '../datasets/Images_split_ttv/train',
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '../datasets/Images_split_ttv/val',
    target_size=(64, 64),
    batch_size=128,
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
nn.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(32, 3, 3, activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dense(16, activation='relu'))
nn.add(Dense(67, activation='softmax'))

# Compile the NN
# opt = optimizers.Adam(lr=0.001)
nn.build(input_shape)
nn.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(nn.summary())

es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
# Start training
history = nn.fit_generator(
    train_generator,
    steps_per_epoch=12467 / 128,
    epochs=120,
    validation_data=validation_generator,
    validation_steps=1532 / 128,
    callbacks=[es])

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
plt.savefig('indoor_cnn_accuracy.pdf')
plt.close()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('indoor_cnn_loss.pdf')

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd

test_generator = test_datagen.flow_from_directory(
    '../datasets/Images_split_ttv/test',
    target_size=(64, 64),
    batch_size=1621,
    class_mode='categorical')

x_test, y_test = test_generator.next()

# Compute probabilities
Y_pred = nn.predict(x_test)
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
# Plot statistics
print('Analysis of results')

path = '../datasets/Images_split_ttv'
classes = {}
for root, dirs, files in os.walk(path, topdown=True):

    path_parts = root.split(os.sep)

    for file in files:
        if path_parts[-1] in classes:
            classes[path_parts[-1]] += 1
        else:
            classes[path_parts[-1]] = 1

target_names = list(classes.keys())
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
df_cm = pd.DataFrame(cm, target_names, target_names)
sn_plot = sn.heatmap(df_cm)
sn_plot.figure.savefig('indoor_confusion_matrix.png')
plt.show()
