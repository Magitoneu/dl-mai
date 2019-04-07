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


def review_to_words(raw_review):
    """
    Only keeps ascii characters in the review and discards @words

    :param raw_review:
    :return:
    """
    letters_only = re.sub("[^a-zA-Z@]", " ", raw_review)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not re.match("^[@]", w)]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--best', help="Save weights best in test", action='store_true', default=False)
    parser.add_argument('--tboard', help="Save log for tensorboard", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

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

    numwords = 200  # Limit the number of words to use
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
    drop = 0.2
    nlayers = 5  # >= 1
    RNN = LSTM  # GRU

    neurons = 256
    embedding = 256

    model = Sequential()
    model.add(Embedding(numwords + 1, embedding, input_length=seq_len))

    if nlayers == 1:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop))
    else:
        model.add(RNN(neurons, implementation=impl, recurrent_dropout=drop, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl, return_sequences=True))
        model.add(RNN(neurons, recurrent_dropout=drop, implementation=impl))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    ############################################
    # Training

    learning_rate = 0.001
    #optimizer = SGD(lr=learning_rate, momentum=0.95)
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    epochs = 50
    batch_size = 100

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

    model.fit(train_x, train_y_c,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks,
              verbose=verbose)

    ############################################
    # Results

    test_y_c = np_utils.to_categorical(test_y, 5)
    score, acc = model.evaluate(test_x, test_y_c,
                                batch_size=batch_size,
                                verbose=verbose)
    print()
    print('Test ACC=', acc)

    test_pred = model.predict_classes(test_x, verbose=verbose)

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
