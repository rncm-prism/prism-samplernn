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
        self.hidden1 = tf.keras.layers.Dense(self.dim, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(self.dim, activation='relu')
        self.outputs = tf.keras.layers.Dense(self.q_levels)

    def call(self, inputs, conditioning_frames):
        batch_size = tf.shape(inputs)[0]

        inputs = self.embedding(tf.reshape(inputs, [-1]))
        inputs = self.inputs(tf.reshape(inputs, [batch_size, -1, self.q_levels]))

        hidden = self.hidden1(inputs + conditioning_frames)
        hidden = self.hidden2(hidden)
        return self.outputs(hidden)
