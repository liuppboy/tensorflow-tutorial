import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import nn
import mnist_utils
import utils

mnist = input_data.read_data_sets('data')

IMAGE_PIXEL = 784
HIDDEN_UNIT = 500
CLASS_NUM = 10
batch_size = 100
leaning_rate = 0.01
max_epoches = 2000

images_placeholder, label_placeholder = mnist_utils.placeholder_inputs()

relu_output = nn.add_layer(images_placeholder, IMAGE_PIXEL, HIDDEN_UNIT, 'hidden')
logits = nn.add_layer_without_act(relu_output, HIDDEN_UNIT, CLASS_NUM, 'softmax_linear')
loss = utils.cross_entrpopy(logits, label_placeholder)
optimizer = tf.train.GradientDescentOptimizer(leaning_rate).minimize(loss)
accuracy = utils.prediction_accuarcy(logits, label_placeholder)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(max_epoches):
        feed_dict = mnist_utils.fill_feed_dict(mnist.train, images_placeholder, label_placeholder, batch_size)
        _, train_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (step+1), "cost=", "{:.9f}".format(train_loss))
    
    print("Accuracy:", accuracy.eval({images_placeholder: mnist.test.images, label_placeholder: mnist.test.labels})) 


