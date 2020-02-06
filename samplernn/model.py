import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode
from .sample_mlp import SampleMLP
from .frame_rnn import FrameRNN

class SampleRNN(tf.keras.layers.Layer):

    def __init__(self, batch_size, frame_sizes, q_levels,
                 dim, n_rnn, seq_len, emb_size):
        super(SampleRNN, self).__init__()
        self.batch_size = batch_size
        self.big_frame_size = frame_sizes[1]
        self.frame_size = frame_sizes[0]
        self.q_levels = q_levels
        self.dim = dim
        self.n_rnn = n_rnn
        self.seq_len = seq_len
        self.emb_size = emb_size

        #self.frame_rnns = [
            #FrameRNN(frame_size, n_rnn=self.n_rnn, dim=self.dim, q_levels=self.q_levels)
            #for frame_size in self.frame_sizes
        #]

        self.big_frame_rnn = FrameRNN(
            frame_size = self.big_frame_size,
            num_lower_tier_frames = self.big_frame_size // self.frame_size,
            n_rnn = self.n_rnn,
            dim = self.dim,
            q_levels = self.q_levels
        )

        self.frame_rnn = FrameRNN(
            frame_size = self.frame_size,
            num_lower_tier_frames = self.frame_size,
            n_rnn = self.n_rnn,
            dim = self.dim,
            q_levels = self.q_levels
        )

        self.sample_mlp = SampleMLP(
            #self.frame_sizes[0], self.dim, self.q_levels, self.emb_size
            self.dim, self.q_levels, self.emb_size
        )

    def call(self, inputs, train_big_frame_state, train_frame_state):
        # UPPER TIER
        (big_frame_outputs, final_big_frame_state) = self.big_frame_rnn(
            tf.cast(inputs, tf.float32)[
                :,
                :-self.big_frame_size,
                :
            ],
            num_steps=(self.seq_len-self.big_frame_size) // self.big_frame_size,
            conditioning_frames=None,
            frame_state=train_big_frame_state,
        )
        # MIDDLE TIER
        (frame_outputs, final_frame_state) = self.frame_rnn(
            tf.cast(inputs, tf.float32)[
                :,
                self.big_frame_size - self.frame_size: -self.frame_size,
                :
            ],
            num_steps=(self.seq_len-self.big_frame_size) // self.frame_size,
            conditioning_frames=big_frame_outputs,
            frame_state=train_frame_state,
        )
        # LOWER TIER (SAMPLES)
        sample_output = self.sample_mlp(
            inputs[
                :,
                self.big_frame_size - self.frame_size: -1,
                :
            ],
            conditioning_frames=frame_outputs,
        )
        return sample_output, final_big_frame_state, final_frame_state