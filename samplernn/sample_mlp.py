import tensorflow as tf


# conv1d = tf.keras.layers.Conv1D(filters=self.dim, kernel_size=frame_size, use_bias=False)
# inputs = tf.keras.layers.Embedding(self.q_levels, self.q_levels)(tf.reshape(data, [-1]))
# inputs = tf.expand_dims(inputs)
# conv1d(inputs)


class SampleMLP(tf.keras.layers.Layer):

    def __init__(self, frame_size, dim, q_levels, emb_size):
        super(SampleMLP, self).__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.emb_size = emb_size
        self.embedding = tf.keras.layers.Embedding(
            self.q_levels, self.q_levels
        )
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.dim, kernel_size=frame_size, use_bias=False
        )

    def build(self, input_shape):
        self.mlp1_weights = self.add_weight(
            name="mlp1_weights",
            shape=[self.dim, self.dim],
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp2_weights = self.add_weight(
            name="mlp2_weights",
            shape=[self.dim, self.dim],
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp3_weights = self.add_weight(
            name="mlp3_weights",
            shape=[self.dim, self.q_levels],
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs, conditioning_frames):
        batch_size = tf.shape(inputs)[0]

        inputs = self.embedding(tf.reshape(inputs, [-1]))
        #inputs = self.conv1d(tf.expand_dims(inputs, 0))
        inputs = self.conv1d(tf.reshape(inputs, [batch_size, -1, self.q_levels]))

        out = inputs + conditioning_frames
        out = tf.reshape(out, [-1, self.dim])
        out = tf.nn.relu(tf.matmul(out, self.mlp1_weights))
        out = tf.nn.relu(tf.matmul(out, self.mlp2_weights))
        out = tf.matmul(out, self.mlp3_weights)

        return tf.reshape(out, [batch_size, -1, self.q_levels])


class SampleMLP_OLD(tf.keras.layers.Layer):

    def __init__(self, dim, q_levels, emb_size):
        super(SampleMLP, self).__init__()
        self.dim = dim
        self.q_levels = q_levels
        self.emb_size = emb_size
        self.embedding2 = tf.keras.layers.Embedding(
            self.q_levels, self.emb_size
        )
        sample_filter_shape = [self.emb_size*2, 1, self.dim]
        initializer = tf.initializers.GlorotNormal()
        self.sample_filter = tf.Variable(
            initializer(shape=sample_filter_shape),
            name="sample_filter",
        )

    def build(self, input_shape):
        initializer = tf.initializers.GlorotUniform()
        self.embedding = tf.Variable(
            initializer(shape=[self.q_levels, self.emb_size])
        )
        self.mlp1_weights = self.add_weight(
            name="mlp1_weights",
            shape=[self.dim, self.dim],
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp2_weights = self.add_weight(
            name="mlp2_weights",
            shape=[self.dim, self.dim],
            dtype=tf.float32,
            trainable=True,
        )
        self.mlp3_weights = self.add_weight(
            name="mlp3_weights",
            shape=[self.dim, self.q_levels],
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
            self.embedding,
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
        out = tf.reshape(out, [-1, self.dim])
        out = tf.nn.relu(tf.matmul(out, self.mlp1_weights))
        out = tf.nn.relu(tf.matmul(out, self.mlp2_weights))
        out = tf.matmul(out, self.mlp3_weights)
        return tf.reshape(out, [-1, sample_shape[1] // self.emb_size - 1, self.q_levels])



