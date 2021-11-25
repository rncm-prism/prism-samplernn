import tensorflow as tf
from .nn import RNN


class FrameRNN(tf.keras.layers.Layer):

    def __init__(self, rnn_type, frame_size, num_lower_tier_frames,
                 num_layers, dim, q_levels, skip_conn, dropout):
        super(FrameRNN, self).__init__()
        self.frame_size = frame_size
        self.num_lower_tier_frames = num_lower_tier_frames
        self.num_layers = num_layers
        self.dim = dim
        self.q_levels = q_levels
        self.skip_conn = skip_conn
        self.inputs = tf.keras.layers.Dense(self.dim)
        self.rnn = RNN(rnn_type, self.dim, self.num_layers, self.skip_conn, dropout=(dropout or 0.0))
        self.upsample = tf.keras.layers.Conv1DTranspose(
            self.dim, self.num_lower_tier_frames, self.num_lower_tier_frames
        )

    def reset_states(self):
        self.rnn.reset_states()

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

        frame_outputs = self.upsample(frame_outputs)

        return frame_outputs
