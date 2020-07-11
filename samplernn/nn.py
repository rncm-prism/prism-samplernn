import tensorflow as tf


# 1D Transposed Convolution, see: https://github.com/tensorflow/tensorflow/issues/6724#issuecomment-357023018
# Also https://github.com/tensorflow/tensorflow/issues/30309#issuecomment-589531625
class Conv1DTranspose(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=1, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__()

    def build(self, input_shape):
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(tf.keras.layers.Conv2DTranspose(
            self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            use_bias=False,
            *self._args, **self._kwargs))
        self._model.add(tf.keras.layers.Lambda(lambda x: x[:,0]))
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)


def rnn_factory(type, *args, **kwargs):
    rnn = getattr(tf.keras.layers, type)
    return rnn(*args, **kwargs)

class RNN(tf.keras.layers.Layer):

    def __init__(self, type, dim, num_layers=1, skip_conn=False, *args, **kwargs):
        self.type = type.upper()
        self.dim = dim
        self.num_layers = num_layers
        self.skip_conn = skip_conn
        self._args, self._kwargs = args, kwargs
        super(RNN, self).__init__()

    def build(self, input_shape):
        self._layer_names = ['layer_' + str(i) for i in range(self.num_layers)]
        for name in self._layer_names:
            self.__setattr__(name, rnn_factory(
            self.type,
            units=self.dim,
            return_sequences=True,
            return_state=True,
            stateful=True,
            *self._args, **self._kwargs))
        if self.skip_conn==True:
            for (i, name) in enumerate(self._layer_names):
                self.__setattr__(name + '_skip_out', tf.keras.layers.Dense(
                    self.dim, kernel_initializer='he_uniform', use_bias=(i==0)))
        super(RNN, self).build(input_shape)

    def run_rnn(self, rnn_name, inputs, state):
        rnn = self.__getattribute__(rnn_name)
        if self.type == 'GRU':
            return rnn(inputs, initial_state=state)
        elif self.type == "LSTM":
            (seqs, state_h, state_c) = rnn(inputs, initial_state=state)
            return (seqs, (state_h, state_c))

    def call(self, inputs):
        seqs = inputs
        state = None
        if not self.skip_conn:
            for name in self._layer_names:
                (seqs, state) = self.run_rnn(name, seqs, state)
            return seqs
        else:
            out = tf.zeros(self.dim)
            for (i, name) in enumerate(self._layer_names):
                seqs = seqs if i==0 else tf.concat((seqs, inputs), axis=2)
                (seqs, state) = self.run_rnn(name, seqs, state)
                dense = self.__getattribute__(name + '_skip_out')
                out += dense(seqs)
            return out