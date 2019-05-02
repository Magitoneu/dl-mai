from keras.applications.inception_v3 import InceptionV3
from keras import models, layers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras import Model, regularizers
from sklearn.decomposition import PCA
from math import pi
from math import ceil


def load_data(path, n_classes=None):
    img_size = (299, 299)
    rands = []
    if n_classes is not None:
        dirs = os.listdir(path)
        rands = np.random.randint(0, len(dirs)-2, n_classes)
    x = []
    y = []
    i = 0
    print('Loading dataset...')
    for root, dirs, files in os.walk(path, topdown=True):
        path_parts = root.split(os.sep)
        inside = False
        for file in files:
            inside = True
            if (i in rands) or n_classes is None:
                img = Image.open(os.path.join(root, file))
                img = np.asarray(img.resize(img_size))
                if img.shape == (img_size[0], img_size[1], 3):
                    x.append(img)
                    y.append(path_parts[-1])
        i = i + (1 if inside else 0)
    indexer = {c: i for i, c in enumerate(np.unique(y))}
    unindexer = {i: c for i, c in enumerate(np.unique(y))}
    print('Dataset loaded')
    print('Categories :', np.unique(y))
    return np.asarray(x), np_utils.to_categorical([indexer[i] for i in y], len(indexer)), unindexer


def build_model(n_classes):
    """
    Function that build the network, it uses the inceptionV3 and the pretrained weights from imagenet.
    And adds on the top of it a dense layer of 128 neurons which will be used as the embedding and a softmax layer to
    classify.
    """
    model = models.Sequential()
    inception_v3 = InceptionV3(include_top=True, weights='imagenet', classes=1000, input_shape=(299, 299, 3))
    print(inception_v3.summary())

    intermediate_model = Model(input=inception_v3.input, outputs=inception_v3.get_layer('avg_pool').output)
    return None, intermediate_model

    # Freeze the entire Inception and just tune the two Dense layer added on the top
    inception_v3.trainable = False
    model.add(inception_v3)
    model.add(layers.Flatten())
    # 128 dimensional embedding
    emb_layer = layers.Dense(256, activation='relu', name='embedding', kernel_regularizer=regularizers.l2())
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
    else:
        N = len(embedding)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.xticks(angles[:-1], list(range(0, N)), color='grey', size=8)

    ax.set_rlabel_position(0)
    ticks = list(range(ceil(min_v-1), ceil(max_v), (ceil(max_v) - ceil(min_v-1)) // 5))
    plt.yticks(ticks, [str(t) for t in ticks], color="grey", size=7)
    plt.ylim(ceil(min_v-1), ceil(max_v))

    for emb in embedding if len(embedding.shape) > 1 else [embedding]:

        emb = np.append(emb, emb[:1])
        ax.plot(angles, emb, linewidth=1, linestyle='solid')
        ax.fill(angles, emb, 'b', alpha=0.1)
    ax.set_title(label)


def show_embeddings(embeddings, labels, n_classes=8):
    """
    Function to plot the embeddings. If the embeddings are a list of lists it will plot different spider plots
    with multiple representations, otherwise it will just plot one representation class.
    """
    min_v = np.min(embeddings)
    max_v = np.max(embeddings)
    times = 1 + (n_classes // 8)
    print('times :', times, n_classes)
    for t in range(0, times):
        for i in range(0, min(len(embeddings) - 8*t, 8)):
            ax = plt.subplot(4, 2, i+1, polar=True)
            spider_embedding(embeddings[i+(t*8)], labels[i+(t*8)], ax, min_v, max_v)
        plt.show()


def dimensionality_reduction(embeddings, embeddings_unseen, n_components=8):
    """
    Function to reduce the dimensolanity of the embeddings in order to make them readable as a plot
    """
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    embeddings_reduced = pca.transform(embeddings)
    embeddings_reduced_unseen = pca.transform(embeddings_unseen)
    print('Total variance explained using %i components: %.4f' % (n_components, np.sum(pca.explained_variance_ratio_)))
    return embeddings_reduced, embeddings_reduced_unseen


def save_plots(history):
    # Acc plot
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


def compute_embeddings(embedding_model, xs, ys, unindexer, x_unseen, y_unseen, unindexer_unseen):
    embeddings = np.asarray(embedding_model.predict(xs))
    embeddings_unseen = np.asarray(embedding_model.predict(x_unseen))
    emb_reduced, emb_reduced_unseen = dimensionality_reduction(embeddings, embeddings_unseen, n_components=8)

    emb_mean = []
    emb_choose = []
    ys = np.array([unindexer[np.argmax(y)] for y in ys])
    for label in np.unique(ys):
        emb_class = emb_reduced[np.argwhere(ys == label).flatten()]
        emb_mean.append(np.mean(emb_class, axis=0))
        chosen = np.random.choice(list(range(0, len(emb_class))), 4)
        emb_choose.append(np.asarray(emb_class)[chosen])

    y_unseen = np.array([unindexer_unseen[np.argmax(y)] for y in y_unseen])
    for label in np.unique(y_unseen):
        emb_class = emb_reduced_unseen[np.argwhere(y_unseen == label).flatten()]
        emb_mean.append(np.mean(emb_class, axis=0))
        chosen = np.random.choice(list(range(0, len(emb_class))), 4)
        emb_choose.append(np.asarray(emb_class)[chosen])

    show_embeddings(emb_mean, np.append(np.unique(ys), np.unique(y_unseen)), n_classes=len(emb_mean))
    show_embeddings(emb_choose, np.append(np.unique(ys), np.unique(y_unseen)), n_classes=len(emb_mean))


if __name__ == '__main__':
    # Prepare dataset
    path_natural = '/home/magi/mai/s2/dl/lab/datasets/natural_images'
    path_emb_test = '/home/magi/mai/s2/dl/lab/datasets/101_ObjectCategories/'
    x, y, unindexer = load_data(path_natural)
    x_unseen, y_unseen, unindexer_unseen = load_data(path_emb_test, n_classes=2)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1337)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1337)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=25,
        brightness_range=(0.6, 1.4),
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Model architecture
    batch_size = 64
    epochs = 25
    model, embedding_model = build_model(len(np.unique(y_train, axis=0)))
    #optimizer = Adam(lr=0.001)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    #es = EarlyStopping(monitor='val_loss', patience=8, mode='min', restore_best_weights=True)
    #history = model.fit_generator(generator=train_datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
    #                              steps_per_epoch=(len(x_train) // batch_size), verbose=1, callbacks=[es],
    #                              validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    #                              validation_steps=len(x_val) // batch_size)
#
    #weights_file = "./weights/natural_emb_model_20epochs" + ".hdf5"
    #model.save_weights(weights_file, overwrite=True)
#
    ## Plots
    #save_plots(history)

    # Test
    test_generator = test_datagen.flow(x_test, y_test, batch_size=len(x_test))
    x_test, y_test = test_generator.next()
    #Y_pred = model.predict(x_test)
    #y_pred = np.argmax(Y_pred, axis=1)

    #results = classification_report(np.argmax(y_test, axis=1), y_pred, output_dict=True)
    #print(classification_report(np.argmax(y_test, axis=1), y_pred))
    #print('Test accuracy: ', np.count_nonzero(y_pred == np.argmax(y_test, axis=1)) / len(y_pred))

    # Embeddings representation
    test_generator = test_datagen.flow(x_unseen, y_unseen, batch_size=len(x_unseen))
    x_unseen, y_unseen = test_generator.next()
    compute_embeddings(embedding_model, x_test, y_test, unindexer, x_unseen, y_unseen, unindexer_unseen)
