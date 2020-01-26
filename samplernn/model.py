import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from .ops import mu_law_encode
from .sample_mlp import SampleMLP
from .frame_rnn import FrameRNN

BIG_FRAME_SIZE = 8
FRAME_SIZE = 2

class SampleRNN(tf.keras.layers.Layer):

    def __init__(self, batch_size, frame_sizes, q_levels,
                 dim, n_rnn, seq_len, emb_size):
        super(SampleRNN, self).__init__()
        self.batch_size = batch_size
        self.frame_sizes = frame_sizes
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
            frame_size=BIG_FRAME_SIZE,
            n_rnn=self.n_rnn,
            dim=self.dim,
            q_levels=self.q_levels
        )

        self.frame_rnn = FrameRNN(
            frame_size=FRAME_SIZE,
            n_rnn=self.n_rnn,
            dim=self.dim,
            q_levels=self.q_levels
        )

        self.sample_mlp = SampleMLP(
            #self.frame_sizes[0], self.dim, self.q_levels, self.emb_size
            self.dim, self.q_levels, self.emb_size
        )

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        encoded = tf.one_hot(
            input_batch,
            depth=self.q_levels,
            dtype=tf.float32,
        )
        shape = [self.batch_size, -1, self.q_levels]
        return tf.reshape(encoded, shape)

    def call(self, inputs, train_big_frame_state, train_frame_state):
        # UPPER TIER
        big_frame_outputs, final_big_frame_state = self.big_frame_rnn(
            tf.cast(inputs, tf.float32)[
                :,
                :-BIG_FRAME_SIZE,
                :
            ],
            num_steps=(self.seq_len-BIG_FRAME_SIZE) // BIG_FRAME_SIZE,
            conditioning_frames=None,
            frame_state=train_big_frame_state,
        )
        # MIDDLE TIER
        frame_outputs, final_frame_state = self.frame_rnn(
            tf.cast(inputs, tf.float32)[
                :,
                BIG_FRAME_SIZE - FRAME_SIZE: -FRAME_SIZE,
                :
            ],
            num_steps=(self.seq_len-BIG_FRAME_SIZE) // FRAME_SIZE,
            conditioning_frames=big_frame_outputs,
            frame_state=train_frame_state,
        )
        # LOWER TIER (SAMPLES)
        sample_output = self.sample_mlp(
            inputs[
                :,
                BIG_FRAME_SIZE - FRAME_SIZE: -1,
                :
            ],
            conditioning_frames=frame_outputs,
        )
        return sample_output, final_big_frame_state, final_frame_state