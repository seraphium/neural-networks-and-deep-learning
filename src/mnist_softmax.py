import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))

#output
y = tf.nn.softmax(tf.matmul(x, w) + b)

#correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

