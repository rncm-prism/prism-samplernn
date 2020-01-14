import tensorflow as tf


class FrameRNN(tf.keras.layers.Layer):

    def __init__(self, n_rnn, dim):
        super(FrameRNN, self).__init__()
        self.dim = dim
        self.n_rnn = n_rnn
        # https://github.com/tensorflow/tensorflow/issues/28216
        cells = [
            tf.keras.layers.GRUCell(dim)
            for _ in range(n_rnn)
        ]
        self.cell = tf.keras.layers.StackedRNNCells(cells)

    def build(self):

  
    def call(self):