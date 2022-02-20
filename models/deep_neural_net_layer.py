"""
Author: Junbong Jang
Date: 7/10/2020

Store keras custom layers
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant
if tf.__version__.split('.')[0] == '2':
    import tensorflow_addons as tfa


# https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
class CustomMultiLossLayer(Layer):
    def __init__(self, removed_tasks, nb_outputs, **kwargs):
        self.removed_tasks = removed_tasks
        self.nb_outputs = nb_outputs
        super().__init__(**kwargs)

    def build(self, input_shape=None):
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]  # each log_var's shape is 1
        super().build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        total_loss = 0
        print('removed_tasks:', self.removed_tasks)
        for i, (y_true, y_pred, log_var) in enumerate(zip(ys_true, ys_pred, self.log_vars)):
            precision = K.exp(-log_var[0])
            if i == 0:  # segmentation
                loss = K.binary_crossentropy(y_true, y_pred) # y_pred shape is: (None,1,196,196)
                if 'seg' in self.removed_tasks:
                    loss = 0
                else:
                    loss = precision * K.mean(loss) + K.abs(log_var[0])
                # tf.print('\nsegmentation', K.mean(loss))
            elif i == 1:  # auto encoder
                loss = tf.keras.losses.mean_squared_error(y_true, y_pred)  # y_pred shape is: (None,3,256,256)
                if 'aut' in self.removed_tasks:
                    loss = 0
                else:
                    loss = precision * K.mean(loss) + K.abs(log_var[0])
            elif i == 2:  # regression
                loss = tf.keras.losses.mean_absolute_error(y_true, y_pred) # y_pred shape is (None,1)
                if 'reg' in self.removed_tasks:
                    loss = 0
                else:
                    loss = precision * K.mean(loss) + K.abs(log_var[0])
                # tf.print('regression', K.mean(loss))
            elif i == 3:  # classification
                loss = tfa.losses.sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.5)  # y_pred shape is  (None,1)
                if 'cls' in self.removed_tasks:
                    loss = 0
                else:
                    loss = K.mean(loss)
                # tf.print('classification', K.mean(loss))
            else:
                raise Exception('i is too big')
            total_loss += loss  # loss shape is (None,)

        return total_loss

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'removed_tasks': self.removed_tasks,
            'nb_outputs': self.nb_outputs
        })
        return config


class CustomLossLayer(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape=None):
        super().build(input_shape)

    def custom_loss(self, y_true, y_pred):
        return tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)(y_true, y_pred)

    def call(self, y_true, y_pred):
        loss = self.custom_loss(y_true, y_pred)
        self.add_loss(loss, inputs=[y_true, y_pred])
        # We won't actually use the output.
        return y_pred