import tensorflow as tf
import cnn
import mnist_utils
import utils
import nn
from tensorflow.examples.tutorials.mnist import input_data

config = utils.Config()
mnist = input_data.read_data_sets(config.train_dir, one_hot=config.is_one_hot)
images_placeholder, labels_placeholder = mnist_utils.placeholder_inputs(config.is_one_hot)

x = tf.reshape(images_placeholder, [-1, 28, 28, 1])
conv1 = cnn.conv2d(x, [5, 5, 1, 32], name='conv1')
max_pool1 = cnn.max_pool2d(conv1, name='max_pool1')
conv2 = cnn.conv2d(max_pool1, [5, 5, 32, 64], name='conv2')
max_pool2 = cnn.max_pool2d(conv2, name='max_pool2')
pool_shape = max_pool2.get_shape().as_list()
input_dim = pool_shape[1] * pool_shape[2] * pool_shape[3]
pool_reshape = tf.reshape(max_pool2, shape=[-1, input_dim])
hidden = nn.add_layer(pool_reshape, input_dim, 512, 'fcl_1')
logits = nn.add_layer_without_act(hidden, 512, 10, 'softmax_linear')
loss = utils.cross_entrpopy(logits, labels_placeholder)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(config.learning_rate, 
                                           global_step=global_step, 
                                           decay_steps=100, 
                                           decay_rate=0.95,
                                           staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
accuracy = utils.prediction_accuarcy(logits, labels_placeholder)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(config.max_iters):
        feed_dict = mnist_utils.fill_feed_dict(mnist.train, images_placeholder, labels_placeholder, config.batch_size)
        _, train_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (step+1), "cost=", "{:.9f}".format(train_loss))
    
    print("Accuracy:", accuracy.eval({images_placeholder: mnist.test.images, labels_placeholder: mnist.test.labels})) 
