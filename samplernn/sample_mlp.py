import tensorflow as tf


class SampleMLP(tf.keras.layers.Layer):

    def __init__(self, frame_size, dim, q_levels, emb_size):
        super(SampleMLP, self).__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.emb_size = emb_size
        self.embedding = tf.keras.layers.Embedding(
            self.q_levels, self.q_levels
        )
        self.inputs = tf.keras.layers.Conv1D(
            filters=self.dim, kernel_size=frame_size, use_bias=False
        )
        self.hidden = tf.keras.layers.Conv1D(
            filters=self.dim, kernel_size=1
        )
        self.outputs = tf.keras.layers.Conv1D(
            filters=self.q_levels, kernel_size=1
        )

    def call(self, inputs, conditioning_frames):
        batch_size = tf.shape(inputs)[0]

        inputs = self.embedding(tf.reshape(inputs, [-1]))
        inputs = self.inputs(tf.reshape(inputs, [batch_size, -1, self.q_levels]))

        out = inputs + conditioning_frames
        out = tf.nn.relu(self.hidden(out))
        return tf.nn.relu(self.outputs(out))
