import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

"""
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10

net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=20*12*12, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 60, mini_batch_size, 0.1,
        validation_data, test_data)
"""

import mnist_loader
import network
print("run basic SGD algo");
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30,  10])
net.SGD(training_data,  30, 10, 0.1, test_data=test_data)

