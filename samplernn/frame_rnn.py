import tensorflow as tf


class FrameRNN(tf.keras.layers.Layer):

    def __init__(self, frame_size, n_rnn, dim, q_levels):
        super(FrameRNN, self).__init__()
        self.frame_size = frame_size
        self.n_rnn = n_rnn
        self.dim = dim
        self.q_levels = q_levels
        # https://github.com/tensorflow/tensorflow/issues/28216
        cells = [
            tf.keras.layers.GRUCell(self.dim)
            for _ in range(n_rnn)
        ]
        self.rnn = tf.keras.layers.StackedRNNCells(cells)

    def build(self, input_shape):
        self.frame_proj_weights = self.add_weight(
            name="frame_proj_weights",
            shape=[
                self.dim,
                self.dim * self.frame_size,
            ],
            dtype=tf.float32,
            trainable=True,
        )
        self.frame_cell_proj_weights = self.add_weight(
            name="frame_cell_proj_weights",
            shape=[
                self.frame_size,
                self.dim,
            ],
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs, num_steps, conditioning_frames, frame_state):
        input_frames = tf.reshape(inputs, [
            tf.shape(inputs)[0],
            tf.shape(inputs)[1] // self.frame_size,
            self.frame_size
        ])
        input_frames = (input_frames / self.q_levels/2.0) - 1.0
        input_frames *= 2.0

        frame_outputs = []

        for time_step in range(num_steps):
            cell_input = tf.reshape(
                input_frames[:, time_step, :], [-1, self.frame_size])
            cell_input = tf.matmul(
                cell_input, self.frame_cell_proj_weights)

            if conditioning_frames is not None:
                cell_input = cell_input + tf.reshape(
                    conditioning_frames[:, time_step, :], [-1, self.dim]
                )

            (frame_cell_output, frame_state) = self.rnn(cell_input, frame_state)
            frame_outputs.append(
                tf.matmul(frame_cell_output, self.frame_proj_weights)
            )

        final_frame_state = frame_state
        frame_outputs = tf.stack(frame_outputs)
        frame_outputs = tf.transpose(frame_outputs, perm=[1, 0, 2])

        frame_outputs = tf.reshape(frame_outputs,
                                   [tf.shape(frame_outputs)[0],
                                    tf.shape(frame_outputs)[1] * self.frame_size,
                                    -1])
        return frame_outputs, final_frame_state