"""
.. module:: SentimentTwAir

SentimentTwAir
*************

:Description: SentimentTwAir


:Version: 

:Created on: 07/09/2017 9:05 

"""

import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from collections import Counter
import argparse
import time
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file
    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def review_to_words(raw_review):
    """
    Only keeps ascii characters in the review and discards @words

    :param raw_review:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", str(raw_review))
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return " ".join(meaningful_words)


def architecture(neurons, drop, nlayers, rnntype, embedding, numwords, impl=1):
    """
    RNN architecture
    :return:
    """
    RNN = LSTM if rnntype == 'LSTM' else GRU
    model = Sequential()
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len))
    if nlayers == 1:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop))
    else:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, return_sequences=True))
        for ii in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

    model.add(Dense(5))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    parser.add_argument('--config', default='config', help='Experiment configuration')
    args = parser.parse_args()

    config = load_config_file(args.config)

    verbose = 1 if args.verbose else 0

    print("Starting:", time.ctime())

    ############################################
    # Data

    review = pandas.read_csv("../datasets/amazon_alexa.tsv", delimiter='\t')
#    review = pandas.read_csv("Presidential.csv")

    # Pre-process the review and store in a separate column
    review['clean_review'] = review['verified_reviews'].apply(lambda x: review_to_words(x))
    review['rating'] = review['rating'].apply(lambda x: x - 1)
    print(review.head())

    # Join all the words in review to build a corpus
    all_text = ' '.join(review['clean_review'])
    words = all_text.split()

    # Convert words to integers
    counts = Counter(words)

    numwords = config['arch']['nwords']  # Limit the number of words to use
    vocab = sorted(counts, key=counts.get, reverse=True)[:numwords]
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    review_ints = []
    for each in review['clean_review']:
        review_ints.append([vocab_to_int[word] for word in each.split() if word in vocab_to_int])

    # Create a list of labels
    labels = np.array(review['rating'])

    # Find the number of reviews with zero length after the data pre-processing
    review_len = Counter([len(x) for x in review_ints])
    print("Zero-length reviews: {}".format(review_len[0]))
    print("Maximum review length: {}".format(max(review_len)))

    # Remove those reviews with zero length and its corresponding label
    review_idx = [idx for idx, review in enumerate(review_ints) if len(review) > 0]
    labels = labels[review_idx]
    review = review.iloc[review_idx]
    review_ints = [review for review in review_ints if len(review) > 0]

    seq_len = max(review_len)
    features = np.zeros((len(review_ints), seq_len), dtype=int)
    for i, row in enumerate(review_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]

    test_size = 0.2
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=test_size)

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    print("Train set: \t\t{}".format(train_y.shape),
          "\nTest set: \t\t{}".format(test_y.shape))

    ############################################
    # Model
    model = architecture(neurons=config['arch']['neurons'],
                         drop=config['arch']['drop'],
                         nlayers=config['arch']['nlayers'],
                         rnntype=config['arch']['rnn'],
                         embedding=config['arch']['emb'],
                         numwords=numwords, impl=2)

    ############################################
    # Training

    learning_rate = config['training']['lrate']
    momentum = config['training']['momentum']
    if config['training']['optimizer'] == 'sgd':
        optimizer = SGD(lr=learning_rate, momentum=momentum)
    elif config['training']['optimizer'] == 'adam':
        optimizer = Adam(lr=learning_rate)
    else:
        raise NameError('Bad optimizer')
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = config['training']['epochs']
    batch_size = config['training']['batch']

    train_y_c = np_utils.to_categorical(train_y, 5)

    callbacks = []
    if args.best:
        modfile = './model%d.h5' % int(time.time())
        m_ckpt = ModelCheckpoint(filepath=modfile, monitor='val_loss', verbose=verbose, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
        callbacks.append(m_ckpt)

    if args.tboard:
        tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))
        callbacks.append(tensorboard)

    print("Start training")
    history = model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks,
              verbose=verbose)

    ############################################
    # Results
    results_name = '_'.join([str(v) for v in list(config['arch'].values())] + [str(v) for v in list(config['training'].values())])
    results_file = results_name + '.txt'

    # Accuracy plot
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(results_name + '_acc.pdf')
    plt.close()
    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(results_name + '_loss.pdf')

    test_y_c = np_utils.to_categorical(test_y, 5)
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,
                                verbose=verbose)

    test_pred = model.predict_classes(test_x, verbose=verbose)

    with open(results_file, 'w') as file:
        file.write("acc: %.3f\n\n" % acc)
        file.write('Confusion Matrix')
        cm = confusion_matrix(test_y, test_pred)
        comma = ","
        s = ''
        for row in cm:
            ss = [str(v) for v in row]
            l = comma.join(ss)
            s = s + '\n' + l
        file.write(s + '\n')
        cr = classification_report(test_y, test_pred, output_dict=True)
        average = list(cr['weighted avg'].values())
        average = ["%.3f" % v for v in average]
        file.write("\nWeighted average: \n" + comma.join(list(cr['weighted avg'].keys())) + '\n')
        file.write(comma.join(average) + '\n')
        file.write('\nClassification Report\n')
        file.write(str(classification_report(test_y, test_pred)))

    print()
    print('Test ACC=', acc)

    print()
    print('Confusion Matrix')
    print('-'*20)
    print(confusion_matrix(test_y, test_pred))
    print()
    print('Classification Report')
    print('-'*40)
    print(classification_report(test_y, test_pred))
    print()
    print("Ending:", time.ctime())
