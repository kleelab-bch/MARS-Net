from tensorflow.keras import backend as K
import tensorflow as tf


def dice_coef(y_true, y_pred):
    smooth = 1.

    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    # average across batch axis, 0-dimension
    dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

    return dice


def f1(y_true, y_pred):
    # referenced https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall + K.epsilon()))
       
       
def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss
    

def balanced_cross_entropy(beta):
    print('balanced_cross_entropy')
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        '''
        beta = tf.reduce_sum(1 - y_true) / (64 * 128 * 128)
        '''
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1 - beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss * (1 - beta))

    return loss
    
    
def temporal_cross_entropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred_logits = tf.math.log(y_pred / (1 - y_pred))  # convert to logits, which is an inverse of sigmoid function
    
    # logits and labels of shape [batch_size, num_classes]
    binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
    binary_ce_loss = tf.reduce_mean(binary_ce_loss, axis=1)

    # measure similarity between frame n with n-1 and n+1
    n_minus_1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred[:,1,:,:], logits=y_pred_logits[:,0,:,:])
    n_plus_1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_pred[:,1,:,:], logits=y_pred_logits[:,2,:,:])
    
    # combine two losses
    n_similarity_loss = n_minus_1_loss * 0.5 + n_plus_1_loss * 0.5
    loss = binary_ce_loss * 0.5 + n_similarity_loss * 0.5
    
    reduced_loss = tf.reduce_mean(loss)
    '''
    tf.print("ypred @@@@@@@@@@@@@@@", y_pred)
    tf.print("ypred_logits @@@@@@@@@@@@@@@", y_pred_logits)
    tf.print("ytrue @@@@@@@@@@@@@@@", y_true)
    tf.print("binary_ce_loss @@@@@@@@@@@@@@@", binary_ce_loss.shape, binary_ce_loss)
    tf.print("n_similarity_loss @@@@@@@@@@@@@@@", n_similarity_loss.shape, n_similarity_loss)
    tf.print("loss @@@@@@@@@@@@@@@", loss.shape, loss)
    tf.print("reduced_loss @@@@@@@@@@@@@@@", reduced_loss)
    '''
    
    return reduced_loss


def zero_loss(y_true, y_pred):
    return 0