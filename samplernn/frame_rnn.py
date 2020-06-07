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


class RNN(tf.keras.layers.Layer):

    def __init__(self, dim, num_layers=1, *args, **kwargs):
        super(RNN, self).__init__()
        self.dim = dim
        self.num_layers = num_layers

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

    def call(self, inputs):
        seqs = inputs
        state = None
        for name in self._layer_names:
            rnn = self.__getattribute__(name)
            (seqs, state) = rnn(seqs, state)
        return seqs


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
