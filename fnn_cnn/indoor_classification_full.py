from __future__ import division
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from keras import optimizers
import numpy as np
import sys
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image
import scipy.io
import pickle as pkl
from keras.initializers import glorot_normal

print('Using Keras version', keras.__version__)

x_train, y_train = [], []
x_test, y_test = [], []
x_val, y_val = [], []

img_size = (128, 128)
batch_size = 32
cache_path = '/fileserver1/magi-nlp/Images_split_ttv'
cache_bin = os.path.join(cache_path, 'images_' + str(img_size) + '.pkl')

if os.path.exists(cache_bin):
    print('Loading from cache')
    x_train, y_train, x_test, y_test, x_val, y_val = pkl.load(open(cache_bin, 'rb'))

else:
    print('Building dataset')
    for root, dirs, files in os.walk('/fileserver1/magi-nlp/Images_split_ttv', topdown=True):

        path_parts = root.split(os.sep)
        if path_parts[-1] != 'Images_split_ttv':
            for file in files:
                img = Image.open(os.path.join(root, file))
                img = np.asarray(img.resize(img_size))
                if img.shape == (img_size[0], img_size[1], 3):
                    if path_parts[-2] == 'train':
                        x_train.append(img)
                        y_train.append(path_parts[-1])
                    elif path_parts[-2] == 'test':
                        x_test.append(img)
                        y_test.append(path_parts[-1])
                    else:
                        x_val.append(img)
                        y_val.append(path_parts[-1])

    indexer = {c: i for i, c in enumerate(np.unique(y_train))}
    unindexer = {i: c for i, c in enumerate(np.unique(y_train))}
    y_train = np_utils.to_categorical([indexer[i] for i in y_train], len(indexer))
    y_test = np_utils.to_categorical([indexer[i] for i in y_test], len(indexer))
    y_val = np_utils.to_categorical([indexer[i] for i in y_val], len(indexer))
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_val = np.asarray(x_val)

    pkl.dump((x_train, y_train, x_test, y_test, x_val, y_val), open(cache_bin, 'wb'))

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=25,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_datagen.fit(x_train)
val_datagen.fit(x_val)
test_datagen.fit(x_test)

# resolution
img_rows, img_cols, channels = img_size[0], img_size[1], 3

if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, channels)

# Define the NN architecture
nn = Sequential()
nn.add(Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='SAME', input_shape=input_shape, kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='SAME', kernel_initializer=glorot_normal()))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dense(512, activation='relu', kernel_initializer=glorot_normal()))
nn.add(Dropout(0.4))
nn.add(Dense(67, activation='softmax', kernel_initializer=glorot_normal()))

# Compile the NN
#opt = optimizers.Adam(lr=0.001)
opt = optimizers.SGD(lr=0.01)
nn.build(input_shape)
nn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(nn.summary())

es = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
# Start training
print(len(x_train), x_train.shape)
history = nn.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=2 * (len(x_train) // batch_size),
    epochs=150,
    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    validation_steps=len(x_val) // batch_size,
    callbacks=[es])

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

test_generator = test_datagen.flow(x_test, y_test, batch_size=1621)

x_test, y_test = test_generator.next()

Y_pred = nn.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
# Plot statistics
print('Analysis of results')

path = '/fileserver1/magi-nlp/Images_split_ttv'
classes = {}
for root, dirs, files in os.walk(path, topdown=True):

    path_parts = root.split(os.sep)
    if path_parts[-1] != 'Images_split_ttv':
        for file in files:
            if path_parts[-1] in classes:
                classes[path_parts[-1]] += 1
            else:
                classes[path_parts[-1]] = 1

target_names = list(classes.keys())
results = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names, output_dict=True)
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
print('Test accuracy: ', np.count_nonzero(y_pred == np.argmax(y_test, axis=1))/len(y_pred))
with open('results.txt', 'w') as the_file:
    the_file.write(str(results['weighted avg']))
