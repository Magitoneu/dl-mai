#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
from time import time
import matplotlib.pyplot as plt


def model(X, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):
        L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=reuse)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, keep_prob, True)

    with tf.variable_scope('L2', reuse=reuse):
        L2 = tf.layers.conv2d(L1, 128, [3, 3], reuse=reuse)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, keep_prob, True)

    with tf.variable_scope('L2-1', reuse=reuse):
        L2_1 = tf.layers.conv2d(L2, 128, [3, 3], reuse=reuse)
        L2_1 = tf.layers.max_pooling2d(L2_1, [2, 2], [2, 2])
        L2_1 = tf.layers.dropout(L2_1, keep_prob, True)

    with tf.variable_scope('L3', reuse=reuse):
        L3 = tf.contrib.layers.flatten(L2_1)
        L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, keep_prob, True)

    with tf.variable_scope('L4', reuse=reuse):
        L4 = tf.layers.dense(L3, 256, activation=tf.nn.relu)

    with tf.variable_scope('LF', reuse=reuse):
        LF = tf.layers.dense(L4, 10, activation=None)

    return LF

gpu_num = 2

# read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
data = data_input[0]

# data layout changes since output should an array of 10 with probabilities
real_output = N.zeros((N.shape(data[0][1])[0], 10), dtype=N.float)
for i in range(N.shape(data[0][1])[0]):
    real_output[i][data[0][1][i]] = 1.0

# data layout changes since output should an array of 10 with probabilities
real_check = N.zeros((N.shape(data[2][1])[0], 10), dtype=N.float)
for i in range(N.shape(data[2][1])[0]):
    real_check[i][data[2][1][i]] = 1.0

# set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

losses = []
X_A = tf.split(x, gpu_num)
Y_A = tf.split(y_, gpu_num)

# Crossentropy
accuracies = []
for gpu_id in range(gpu_num):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
            print(gpu_id, gpu_id > 0)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                                        labels=Y_A[gpu_id], logits=model(X_A[gpu_id], gpu_id > 0))

            correct_prediction = tf.equal(tf.argmax(model(X_A[gpu_id], True), 1), tf.argmax(Y_A[gpu_id], 1))
            accuracy = tf.cast(correct_prediction, tf.float32)
            losses.append(loss)
            accuracies.append(accuracy)

accuracy = tf.reduce_mean(tf.concat(accuracies, axis=0))
loss = tf.reduce_mean(tf.concat(losses, axis=0))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, colocate_gradients_with_ops=True)

batch_size = 500
acc_history = []
loss_history = []

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    # TRAIN
    print("TRAINING")
    t_start = time()

    for i in range(1000):

        batch_idx = N.random.choice(range(0, len(real_output)), batch_size)

        batch_xs = data[0][0][batch_idx]
        batch_ys = real_output[batch_idx]

        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, loss_v = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        loss_history.append(loss_v)
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        acc_history.append(train_accuracy)
        if i % 10 == 0:
            print('step %d, training accuracy %g' % (i, train_accuracy))

    print('Training time: ', time() - t_start)
    # TEST

    plt.plot(loss_history)
    plt.title('Loss ' + str(gpu_num) + ' GPUs')
    plt.savefig('loss_' + str(gpu_num) + '.png')
    plt.close()

    plt.plot(acc_history)
    plt.title('Accuracy ' + str(gpu_num) + ' GPUs')
    plt.savefig('acc_' + str(gpu_num) + '.png')
    plt.close()

    print("TESTING")
    train_accuracy = accuracy.eval(feed_dict={x: data[2][0].reshape(-1, 28, 28, 1), y_: real_check, keep_prob: 1.0})
    print('test accuracy %g' % (train_accuracy))
