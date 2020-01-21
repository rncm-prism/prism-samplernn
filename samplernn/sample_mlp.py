import tensorflow as tf


class SampleMLP(tf.keras.layers.Layer):

    def __init__(self, dim, q_levels, emb_size):
        super(SampleMLP, self).__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.emb_size = emb_size
        sample_filter_shape = [self.emb_size*2, 1, self.dim]
        initializer = tf.initializers.GlorotNormal(sample_filter_shape)
        self.sample_filter = tf.Variable(sample_filter_shape, name="sample_filter")

    def build(self, input_shape):
        self.mlp1_weights = self.add_weight(
            name="mlp1_weights",
            shape=(input_shape),
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp2_weights = self.add_weight(
            name="mlp2_weights",
            shape=(input_shape),
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp3_weights = self.add_weight(
            name="mlp3_weights",
            shape=(input_shape),
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs, conditioning_frames):
        sample_shape = [
            tf.shape(inputs)[0],
            tf.shape(inputs)[1]*self.emb_size,
            1,
        ]
        inputs = tf.nn.embedding_lookup(
            [self.q_levels, self.emb_size],
            tf.reshape(inputs, [-1]),
        )
        inputs = tf.reshape(inputs, sample_shape)
        out = tf.nn.conv1d(
            inputs,
            self.sample_filter,
            stride=self.emb_size,
            padding="VALID",
            name="sample_conv",
        )
        out = out + conditioning_frames
        out = tf.nn.relu(tf.matmul(out, self.mlp1_weights))
        out = tf.nn.relu(tf.matmul(out, self.mlp2_weights))
        out = tf.matmul(out, self.mlp3_weights)
        return tf.reshape(out, [-1, sample_shape[1] // self.emb_size - 1, self.q_levels])
