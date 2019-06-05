#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt


#read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
#print ( N.shape(data[0][0])[0] )
#print ( N.shape(data[0][1])[0] )

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0



#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

lr = 0.5
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#TRAINING PHASE
print("TRAINING")
loss_history = []
acc_history = []
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for i in range(500):
  batch_xs = data[0][0][100*i:100*i+100]
  batch_ys = real_output[100*i:100*i+100]
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  loss = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
  loss_history.append(loss[0])
  acc_history.append(loss[1])

plt.plot(loss_history)
method = 'Adam ' + str(lr)
plt.title('Loss ' + method)
plt.savefig('/home/magi/mai/s2/dl/lab/hpc/2_loss_' + method + '.png')
plt.show()

plt.plot(acc_history)
plt.title('Accuracy ' + method)
plt.savefig('/home/magi/mai/s2/dl/lab/hpc/2_acc_' + method + '.png')
plt.show()
#CHECKING THE ERROR
print("ERROR CHECK")
print(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))


