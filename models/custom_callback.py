'''
Date 7/8/2021
Junbong Jang

Defines callbacks called in fit() for training
'''
import time
import tensorflow as tf


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class TrainableLossWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.history_of_trainable_loss_weights = []
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(self.get_trainable_loss_weights())
        self.history_of_trainable_loss_weights.append(self.get_trainable_loss_weights())
        # print(self.model.layers[-2].get_weights()[0][0][0][0])

    def get_trainable_loss_weights(self):
        return self.model.layers[-1].get_weights()