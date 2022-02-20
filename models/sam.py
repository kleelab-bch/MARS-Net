'''
Junbong Jang
Date 7/3/2021

Referenced
https://github.com/Jannoshh/simple-sam
https://github.com/sayakpaul/Sharpness-Aware-Minimization-TensorFlow
'''
import tensorflow as tf


class SAMModel(tf.keras.Model):
    def __init__(self, a_model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        print('SAMModel called')
        self.a_model = a_model
        self.rho = rho

    def train_step(self, data):
        (images, labels) = data
        e_ws = []
        with tf.GradientTape() as tape:
            predictions = self.a_model(images)
            loss = self.compiled_loss(labels, predictions)
        trainable_params = self.a_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        with tf.GradientTape() as tape:
            predictions = self.a_model(images)
            loss = self.compiled_loss(labels, predictions)
        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)
        sam_gradients = tape.gradient(loss, trainable_params)
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients(
            zip(sam_gradients, trainable_params))

        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data):
    #     (images, labels) = data
    #     predictions = self.a_model(images, training=False)
    #     loss = self.compiled_loss(labels, predictions)
    #     self.compiled_metrics.update_state(labels, predictions)
    #     return {m.name: m.result() for m in self.metrics}

    def call(self, x):
        return self.a_model(x)

    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm