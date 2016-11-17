import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def placeholder_inputs(is_one_hot=False):
    images_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
    if is_one_hot:
        labels_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
    else:
        labels_placeholder = tf.placeholder(tf.int32, shape=[None])
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl, batch_size, is_one_hot=False):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(batch_size, is_one_hot)
    feed_dict = {images_pl: images_feed, labels_pl: labels_feed,}
    return feed_dict