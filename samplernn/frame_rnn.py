import tensorflow as tf


class FrameRNN(tf.keras.layers.Layer):

    def __init__(self, frame_size, num_lower_tier_frames, n_rnn, dim, q_levels):
        super(FrameRNN, self).__init__()
        self.frame_size = frame_size
        self.num_lower_tier_frames = num_lower_tier_frames
        self.n_rnn = n_rnn
        self.dim = dim
        self.q_levels = q_levels
        self.rnn = tf.keras.layers.GRU(
            self.dim,
            return_sequences=True,
            stateful=True,
            #unroll=True
        )

    def build(self, input_shape):
        initializer = tf.initializers.GlorotNormal()
        self.conv1d = tf.Variable(
            initializer(shape=[1, self.frame_size, self.dim]),
            name="conv1d",
        )
        self.upsample = tf.Variable(
            initializer(shape=[self.num_lower_tier_frames, self.dim, self.dim]),
            name="upsample",
        )

    def call(self, inputs, num_steps, conditioning_frames=None):
        # When running in tf.function mode this type of assignment caused an error
        # (batch_size, _, _) = tf.shape(inputs)
        batch_size = tf.shape(inputs)[0]

        input_frames = tf.reshape(inputs, [
            batch_size,
            tf.shape(inputs)[1] // self.frame_size, #num_steps,
            self.frame_size
        ])
        input_frames = ( (input_frames / (self.q_levels / 2.0)) - 1.0 ) * 2.0

        input_frames = tf.nn.conv1d(
            input_frames,
            self.conv1d,
            stride=1,
            padding="VALID"
        )

        if conditioning_frames is not None:
            input_frames += conditioning_frames

        frame_outputs = self.rnn(input_frames)

        output_shape = [
            batch_size,
            num_steps * self.num_lower_tier_frames,
            self.dim
        ]
        frame_outputs = tf.nn.conv1d_transpose(
            frame_outputs,
            self.upsample,
            strides=self.num_lower_tier_frames,
            output_shape=output_shape,
        )

        return frame_outputs
