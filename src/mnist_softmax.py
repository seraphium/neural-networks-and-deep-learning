import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

b = tf.Variable(tf.zeros(10))
x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))

#output
y = tf.nn.softmax(tf.matmul(x, w) + b)

#correct answer
y_ = tf.placeholder(tf.float32, [None, 10])
'''
cost = y_ * tf.log(y)
sumTrainingCost = -tf.reduce_sum(cost, reduction_indices=[1])
cross_entropy = tf.reduce_mean(sumTrainingCost)
'''
# Regularization parameter
lbda = tf.constant(0.001)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) + lbda * tf.nn.l2_loss(w)
    #tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

epoch = 1000
for index in range(epoch):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    if index % 100 == 0 :
        print("epoch:",  index)
        print("training accuracy:",sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels}))
        print("test accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#print("test accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))