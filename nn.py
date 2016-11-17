import tensorflow as tf
        
def add_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        rand_range = tf.sqrt(6. / float(input_dim + output_dim))
        weights = tf.Variable(tf.random_uniform([input_dim, output_dim],-rand_range, rand_range), name='Weights') 
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_dim]), name='Biases')
        activations = act(tf.matmul(input_tensor, weights) + biases)
        return activations

def add_layer_without_act(input_tensor, input_dim, output_dim, layer_name):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        rand_range = tf.sqrt(6. / float(input_dim + output_dim))
        weights = tf.Variable(tf.random_uniform([input_dim, output_dim],-rand_range, rand_range), name='Weights') 
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[output_dim]), name='Biases')
        logits = tf.matmul(input_tensor, weights) + biases
        return logits

