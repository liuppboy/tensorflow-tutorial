import tensorflow as tf

def conv2d(input, filter_shape, stride=1, act=tf.nn.relu, name='conv'):
    '''
    Args: filter_shape:[ksize, ksize, input_filter_num, output_filter_num]
    '''
    with tf.name_scope(name):
        weights = weights_init(filter_shape)
        biases = biases_init(filter_shape[3])
        output = tf.nn.bias_add(tf.nn.conv2d(
            input, weights, strides=[1, stride, stride, 1], padding='SAME'), biases)
        return act(output)
    
def max_pool2d(input, ksize=2, stride=2, name='max_pool'):
    with tf.name_scope(name):
        output = tf.nn.max_pool(input, ksize=[1,ksize, ksize, 1], 
                                strides=[1, stride, stride, 1], padding='SAME')    
    return output

def ave_pool2d(input, ksize=2, stride=2, name='ave_pool'):
    with tf.name_scope(name):
        output = tf.nn.avg_pool(input, ksize=[1,ksize, ksize, 1], 
                                strides=[1, stride, stride, 1], padding='SAME')    
    return output

def weights_init(shape):
    '''init weights variable in cnn
    Arg: shape [ksize, ksize, input_filter_num, output_filter_num]
    return init filter
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=.1, dtype=tf.float32), name='weights')

def biases_init(filter_num):
    return tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[filter_num]), name='biases')