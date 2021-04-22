import tensorflow as tf


class SquareErrorDirichlet(tf.keras.losses.Loss):
    def __init__(self, name="square_error_dirichlet"):
        super().__init__(name=name)

    def call(self, y_true, alpha):
        y_true = tf.cast(y_true, tf.float32)
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        prob = tf.divide(alpha, strength)
        loss = tf.square(prob - y_true) + prob * (1 - prob) / (strength + 1.0)
        return loss
