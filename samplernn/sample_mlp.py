import tensorflow as tf


class SampleMLP(tf.keras.layers.Layer):

    def __init__(self, dim, q_levels, emb_size):
        super(SampleMLP, self).__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.emb_size = emb_size
        initializer = tf.initializers.GlorotNormal([self.emb_size*2, 1, self.dim])
        self.sample_filter = tf.Variable(initializer, name="sample_filter")

    def build(self, input_shape, units):
        self.mlp1 = self.add_weight(shape=(input_shape, units),
                                initializer='random_normal',
                                trainable=True)
        self.mlp2 = self.add_weight(shape=(input_shape, units),
                                initializer='random_normal',
                                trainable=True)
        self.mlp3 = self.add_weight(shape=(input_shape, units),
                                initializer='random_normal',
                                trainable=True)

    def call(self, frame_outputs, sample_input_sequences):
        sample_shape = [tf.shape(sample_input_sequences)[0],
            tf.shape(sample_input_sequences)[1]*self.emb_size,
            1]
        sample_input_sequences = tf.nn.embedding_lookup(
            [self.q_levels, self.emb_size], tf.reshape(sample_input_sequences, [-1]))
        sample_input_sequences = tf.reshape(
            sample_input_sequences, sample_shape)
        out = tf.nn.conv1d(sample_input_sequences,
                        self.sample_filter,
                        stride=self.emb_size,
                        padding="VALID",
                        name="sample_conv")
        out = out + frame_outputs
        out = tf.nn.relu(tf.matmul(out, self.mlp1))
        out = tf.nn.relu(tf.matmul(out, self.mlp2))
        out = tf.matmul(out, self.mlp3)
        return tf.reshape(out, [-1, sample_shape[1] // self.emb_size - 1, self.q_levels])
