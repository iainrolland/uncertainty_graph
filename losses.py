import tensorflow as tf


class SquareErrorDirichlet(tf.keras.losses.Loss):
    def __init__(self, name="square_error_dirichlet"):
        super().__init__(name=name)

    def call(self, y_true, alpha):
        y_true = tf.cast(y_true, tf.float32)
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        prob = tf.divide(alpha, strength)
        loss = tf.square(prob - y_true) + prob * (1 - prob) / (strength + 1.0)
        return tf.reduce_sum(loss, axis=-1)


class TeacherAndSquareErrorDirichlet(tf.keras.losses.Loss):
    def __init__(self, teacher_prob, name="teacher_and_square_error_dirichlet"):
        super().__init__(name=name)
        self.epochs = 0
        self.teacher_prob = teacher_prob

    def call(self, y_true, alpha):
        y_true = tf.cast(y_true, tf.float32)
        strength = tf.reduce_sum(alpha, axis=1, keepdims=True)
        prob = tf.divide(alpha, strength)
        loss = tf.reduce_sum(tf.square(prob - y_true) + prob * (1 - prob) / (strength + 1.0), axis=-1)
        loss += tf.losses.KLD(self.teacher_prob, prob) * tf.minimum(self.epochs / 200, 1)
        self.epochs += 1
        return loss
