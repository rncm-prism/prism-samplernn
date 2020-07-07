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


class GRU(tf.keras.layers.Layer):

    def __init__(self, dim, num_layers=1, skip_conn=False, *args, **kwargs):
        assert not(num_layers==1 and skip_conn==True), \
            "Only RNNs with more than a single layer can have skip connections."
        super(GRU, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.skip_conn = skip_conn

        def layer():
            return tf.keras.layers.GRU(
                self.dim,
                return_sequences=True,
                return_state=True,
                stateful=True,
                *args, **kwargs)

        self._layer_names = ['layer_' + str(i) for i in range(self.num_layers)]
        for name in self._layer_names:
             self.__setattr__(name, layer())

        if self.skip_conn==True:
            self.out_bias = tf.keras.layers.Dense(
                self.dim, kernel_initializer='he_uniform')
            self.out_no_bias = tf.keras.layers.Dense(
                self.dim, kernel_initializer='he_uniform', use_bias=False)

    def call(self, inputs):
        seqs = inputs
        state = None
        if not self.skip_conn:
            for name in self._layer_names:
                rnn = self.__getattribute__(name)
                (seqs, state) = rnn(seqs, initial_state=state)
            return seqs
        else:
            out = tf.zeros(self.dim)
            for (i, name) in enumerate(self._layer_names):
                rnn = self.__getattribute__(name)
                seqs = seqs if i==0 else tf.concat((seqs, inputs), axis=2)
                (seqs, state) = rnn(seqs, initial_state=state)
                out += self.out_bias(seqs) if i==0 else self.out_no_bias(seqs)
            return out


class LSTM(tf.keras.layers.Layer):

    def __init__(self, dim, num_layers=1, skip_conn=False, *args, **kwargs):
        super(LSTM, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.skip_conn = skip_conn

        def layer():
            return tf.keras.layers.LSTM(
                self.dim,
                return_sequences=True,
                return_state=True,
                stateful=True,
                *args, **kwargs)

        self._layer_names = ['layer_' + str(i) for i in range(self.num_layers)]
        for name in self._layer_names:
             self.__setattr__(name, layer())

        if self.skip_conn==True:
            self.out_bias = tf.keras.layers.Dense(
                self.dim, kernel_initializer='he_uniform')
            self.out_no_bias = tf.keras.layers.Dense(
                self.dim, kernel_initializer='he_uniform', use_bias=False)

    def call(self, inputs):
        seqs = inputs
        state = None
        if not self.skip_conn:
            for name in self._layer_names:
                rnn = self.__getattribute__(name)
                (seqs, state_h, state_c) = rnn(seqs, initial_state=state)
                state = [state_h, state_c]
            return seqs
        else:
            out = tf.zeros(self.dim)
            for (i, name) in enumerate(self._layer_names):
                rnn = self.__getattribute__(name)
                seqs = seqs if i==0 else tf.concat((seqs, inputs), axis=2)
                (seqs, state_h, state_c) = rnn(seqs, initial_state=state)
                state = [state_h, state_c]
                out += self.out_bias(seqs) if i==0 else self.out_no_bias(seqs)
            return out