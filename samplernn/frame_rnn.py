import tensorflow as tf
from .nn import RNN


class FrameRNN(tf.keras.layers.Layer):

    def __init__(self, frame_size, num_lower_tier_frames, num_layers, dim, q_levels):
        super(FrameRNN, self).__init__()
        self.frame_size = frame_size
        self.num_lower_tier_frames = num_lower_tier_frames
        self.num_layers = num_layers
        self.dim = dim
        self.q_levels = q_levels
        self.inputs = tf.keras.layers.Dense(self.dim)
        self.rnn = RNN(self.dim, self.num_layers)

    def build(self, input_shape):
        self.upsample = tf.Variable(
            tf.initializers.GlorotNormal()(
                shape=[self.num_lower_tier_frames, self.dim, self.dim]),
            name="upsample",
        )

    def call(self, inputs, conditioning_frames=None):
        # When running in tf.function mode this type of assignment caused an error
        # (batch_size, _, _) = tf.shape(inputs)
        batch_size = tf.shape(inputs)[0]

        input_frames = tf.reshape(inputs, [
            batch_size,
            tf.shape(inputs)[1] // self.frame_size,
            self.frame_size
        ])
        input_frames = ( (input_frames / (self.q_levels / 2.0)) - 1.0 ) * 2.0
        num_steps = tf.shape(input_frames)[1]

        input_frames = self.inputs(input_frames)

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
