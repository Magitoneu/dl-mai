from keras.applications.inception_v3 import InceptionV3
from keras import models, layers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
import os
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, regularizers
from sklearn.decomposition import PCA
from math import pi
from math import ceil
import sys


def load_data(path):
    img_size = (150, 150)
    x = []
    y = []
    print('Loading dataset...')
    for root, dirs, files in os.walk(path, topdown=True):
        path_parts = root.split(os.sep)
        for file in files:
            img = Image.open(os.path.join(root, file))
            img = np.asarray(img.resize(img_size))
            x.append(img)
            y.append(path_parts[-1])
    indexer = {c: i for i, c in enumerate(np.unique(y))}
    unindexer = {i: c for i, c in enumerate(np.unique(y))}
    print('Dataset loaded')
    return np.asarray(x), np_utils.to_categorical([indexer[i] for i in y], len(indexer)), unindexer


def build_model(n_classes):
    model = models.Sequential()
    inception_v3 = InceptionV3(include_top=False, weights='imagenet', classes=n_classes, input_shape=(150, 150, 3))
    inception_v3.trainable = False
    model.add(inception_v3)
    model.add(layers.Flatten())
    # 128 dimensional embedding
    emb_layer = layers.Dense(128, activation='relu', name='embedding', kernel_regularizer=regularizers.l1_l2())
    model.add(emb_layer)
    model.add(layers.Dense(n_classes, activation='softmax'))

    intermediate_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)

    return model, intermediate_model


def embeddings_clustering(model, x_train, x_test, y_train, y_test):
    pass


def spider_embedding(embedding, label, ax, min_v, max_v):
    """
    Function to get the embedding representation as a spider/radar plot
    """

    if len(embedding.shape) > 1:
        N = len(embedding[0])
        print(embedding)
    else:
        N = len(embedding)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], list(range(0, N)), color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    ticks = list(range(ceil(min_v), ceil(max_v), (ceil(max_v) - ceil(min_v)) // 5))
    plt.yticks(ticks, [str(t) for t in ticks], color="grey", size=7)
    plt.ylim(ceil(min_v), ceil(max_v))

    for emb in embedding if len(embedding.shape) > 1 else [embedding]:

        emb = np.append(emb, emb[:1])
        # Plot data
        ax.plot(angles, emb, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, emb, 'b', alpha=0.1)

    ax.set_title(label)


def show_embeddings(embeddings, labels, n_classes=8):
    min_v = np.min(embeddings)
    max_v = np.max(embeddings)
    n_emb = len(embeddings)
    times = 1 + (n_emb % n_classes)
    for t in range(0, times):
        for i, emb in enumerate(embeddings):
            ax = plt.subplot(4, 2, i+1, polar=True)
            spider_embedding(emb, labels[i], ax, min_v, max_v)
        plt.show()


def dimensionality_reduction(embeddings, n_components=8):
    """
    Function to reduce the dimensolanity of the embeddings in order to make them readable as a plot
    """
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    return embeddings_reduced


def save_plots(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('natural_emb_acc.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('natural_emb_loss.pdf')


if __name__ == '__main__':
    # Prepare dataset
    path = '/home/magi/mai/s2/dl/lab/datasets/natural_images'
    x, y, unindexer = load_data(path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1337)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1337)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Model architecture
    batch_size = 64
    epochs = 10
    model, embedding_model = build_model(len(np.unique(y_train, axis=0)))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
    history = model.fit_generator(generator=train_datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                  steps_per_epoch=(len(x_train) // batch_size), verbose=1, callbacks=[es],
                                  validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
                                  validation_steps=len(x_val) // batch_size)

    weights_file = "./weights/natural_emb_model" + ".hdf5"
    model.save_weights(weights_file, overwrite=True)

    # Accuracy plot
    save_plots(history)

    test_generator = test_datagen.flow(x_test, y_test, batch_size=len(x_test))
    x_test, y_test = test_generator.next()
    Y_pred = model.predict(x_test)
    y_pred = np.argmax(Y_pred, axis=1)

    results = classification_report(np.argmax(y_test, axis=1), y_pred, output_dict=True)
    print(classification_report(np.argmax(y_test, axis=1), y_pred))
    print('Test accuracy: ', np.count_nonzero(y_pred == np.argmax(y_test, axis=1)) / len(y_pred))

    embeddings = np.asarray(embedding_model.predict(x_test))
    emb_reduced = dimensionality_reduction(embeddings, n_components=8)

    emb_mean = []
    emb_choose = []
    y_test = np.array([unindexer[np.argmax(y)] for y in y_test])
    for label in np.unique(y_test):
        emb_class = emb_reduced[np.argwhere(y_test == label).flatten()]
        emb_mean.append(np.mean(emb_class, axis=0))
        chosen = np.random.choice(list(range(0, len(emb_class))), 4)
        emb_choose.append(np.asarray(emb_class)[chosen])

    show_embeddings(emb_mean, np.unique(y_test), len(emb_mean))
    show_embeddings(emb_choose, np.unique(y_test), len(emb_mean))

