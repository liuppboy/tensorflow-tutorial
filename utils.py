import tensorflow as tf

def cross_entrpopy(logits, labels, is_one_hot=False):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, if is_one_hot = True: float - [batch_size, NUM_CLASS]
                             if is_one_hot = False: int32 - [batch_size]
    Returns:
      loss: Loss tensor of type float.
    """
    if is_one_hot:
        sum_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.to_float32(labels))
    else:
        sum_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.to_int32(labels))
    loss = tf.reduce_mean(sum_cross_entropy, name='cross_entropy_loss')
    return loss

def prediction_accuarcy(logits, labels, is_ont_hot=False):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, if is_one_hot = True: float - [batch_size, NUM_CLASS]
                             if is_one_hot = False: int32 - [batch_size]
    Returns:
      loss: Loss tensor of type float.
    """  
    if is_ont_hot:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    else:
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy



