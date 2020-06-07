import tensorflow as tf
from .sample_mlp import SampleMLP
from .frame_rnn import FrameRNN


class SampleRNN(tf.keras.Model):

    def __init__(self, batch_size, frame_sizes, q_levels, q_type,
                 dim, num_rnn_layers, seq_len, emb_size):
        super(SampleRNN, self).__init__()
        self.batch_size = batch_size
        self.big_frame_size = frame_sizes[1]
        self.frame_size = frame_sizes[0]
        self.q_type = q_type
        self.q_levels = q_levels
        self.dim = dim
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.emb_size = emb_size

        self.big_frame_rnn = FrameRNN(
            frame_size = self.big_frame_size,
            num_lower_tier_frames = self.big_frame_size // self.frame_size,
            num_layers = self.num_rnn_layers,
            dim = self.dim,
            q_levels = self.q_levels
        )

        self.frame_rnn = FrameRNN(
            frame_size = self.frame_size,
            num_lower_tier_frames = self.frame_size,
            num_layers = self.num_rnn_layers,
            dim = self.dim,
            q_levels = self.q_levels
        )

        self.sample_mlp = SampleMLP(
            self.frame_size, self.dim, self.q_levels, self.emb_size
        )

    def call(self, inputs):
        # UPPER TIER
        big_frame_outputs = self.big_frame_rnn(
            tf.cast(inputs, tf.float32)[:, : -self.big_frame_size, :]
        )
        # MIDDLE TIER
        frame_outputs = self.frame_rnn(
            tf.cast(inputs, tf.float32)[:, self.big_frame_size-self.frame_size : -self.frame_size, :],
            conditioning_frames=big_frame_outputs,
        )
        # LOWER TIER (SAMPLES)
        sample_output = self.sample_mlp(
            inputs[:, self.big_frame_size - self.frame_size : -1, :],
            conditioning_frames=frame_outputs,
        )
        return sample_output