import tensorflow as tf
import layers


class GCN(tf.keras.Model):

    def __init__(self, n_labels, channels=16, activation="relu", output_activation="softmax", use_bias=False,
                 dropout_rate=0.5, l2_reg=2.5e-4, n_input_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.n_input_channels = n_input_channels
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = layers.GraphConvolution(
            channels, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), use_bias=use_bias
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = layers.GraphConvolution(
            channels, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), use_bias=use_bias
        )
        self._d2 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn2 = layers.GraphConvolution(
            n_labels, activation=output_activation, use_bias=use_bias
        )

        if tf.version.VERSION < "2.2":
            if n_input_channels is None:
                raise ValueError("n_input_channels required for tf < 2.2")
            x = tf.keras.Input((n_input_channels,), dtype=tf.float32)
            a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
            self._set_inputs((x, a))

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            n_input_channels=self.n_input_channels,
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader
        if self.n_input_channels is None:
            self.n_input_channels = x.shape[-1]
        else:
            assert self.n_input_channels == x.shape[-1]
        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        x = self._gcn1([x, a])
        x = self._d2(x)
        x = self._gcn2([x, a])
        return x



