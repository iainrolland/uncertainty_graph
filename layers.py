import tensorflow as tf
from spektral.layers.ops import modal_dot


# noinspection PyAttributeOutsideInit,PyMethodOverriding
class GraphConvolution(tf.keras.layers.Layer):
    """Graph convolution layer."""

    def __init__(self, channels, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(GraphConvolution, self).__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.channels = channels
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2  # require an adjacency matrix of some kind as well as some node features
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs):
        x, a = inputs

        output = tf.keras.backend.dot(x, self.kernel)
        output = modal_dot(a, output)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)
        output = self.activation(output)

        return output
