from keras.applications.inception_v3 import InceptionV3
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
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
    Function that build the network, it uses the inceptionV3 and the pretrained weights from imagenet. The layer
    avg_pool (the one before the predicting layer) is the one used as embedding.

    """
    inception_v3 = InceptionV3(include_top=True, weights='imagenet', classes=1000, input_shape=(299, 299, 3))

    intermediate_model = Model(inputs=inception_v3.input, outputs=inception_v3.get_layer('avg_pool').output)
    return intermediate_model


def embeddings_clustering(embeddings, labels, unindexer):
    labels = np.array([unindexer[np.argmax(y)] for y in labels])
    kmeans = KMeans(n_clusters=len(np.unique(labels)))
    y_pred = kmeans.fit_predict(embeddings)
    print('Adjusted rand score: ', adjusted_rand_score(labels, y_pred))


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


def dimensionality_reduction(embeddings, n_components=8):
    """
    Function to reduce the dimensolanity of the embeddings in order to make them readable as a plot
    """
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    print('Total variance explained using %i components: %.4f' % (n_components, np.sum(pca.explained_variance_ratio_)))
    return embeddings_reduced


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


def represent_embeddings(embeddings, ys, unindexer):
    emb_reduced = dimensionality_reduction(embeddings, n_components=12)

    emb_mean = []
    emb_choose = []
    ys = np.array([unindexer[np.argmax(y)] for y in ys])
    for label in np.unique(ys):
        emb_class = emb_reduced[np.argwhere(ys == label).flatten()]
        emb_mean.append(np.mean(emb_class, axis=0))
        chosen = np.random.choice(list(range(0, len(emb_class))), 4)
        emb_choose.append(np.asarray(emb_class)[chosen])

    show_embeddings(emb_mean, np.unique(ys), n_classes=len(emb_mean))
    show_embeddings(emb_choose, np.unique(ys), n_classes=len(emb_mean))


if __name__ == '__main__':
    # Prepare dataset
    # path_natural = '/home/magi/mai/s2/dl/lab/datasets/natural_images'
    path_caltech = '/home/magi/mai/s2/dl/lab/datasets/101_ObjectCategories/'
    x, y, unindexer = load_data(path_caltech, n_classes=20)

    images_per_class = 50
    x_reduced, y_reduced = [], []

    y_labels = np.array([unindexer[np.argmax(y)] for y in y])
    for label in np.unique(y_labels):
        selected = np.random.choice(np.argwhere(y_labels == label).flatten(), min(images_per_class, np.sum(y_labels == label)))
        x_reduced = (x_reduced + x[selected].tolist() if len(x_reduced) > 0 else x[selected].tolist())
        y_reduced = (y_reduced + y[selected].tolist() if len(y_reduced) > 0 else y[selected].tolist())

    x_reduced = np.asarray(x_reduced)
    y_reduced = np.asarray(y_reduced)
    print(x_reduced.shape, y_reduced.shape)
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow(x_reduced, y_reduced, batch_size=len(y_reduced))
    x_reduced, y_reduced = train_generator.next()

    # Create model
    embedding_model = build_model(len(np.unique(y_reduced, axis=0)))

    # Compute embeddings
    embeddings = np.asarray(embedding_model.predict(x_reduced))

    # Cluster data
    # embeddings_clustering(embeddings, y_reduced, unindexer)

    # Embeddings representation
    represent_embeddings(embeddings, y_reduced, unindexer)
