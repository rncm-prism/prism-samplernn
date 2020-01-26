import numpy as np
import tensorflow as tf
from samplernn import load_audio

# Converting just the output array makes this incredibly slow, but
# also converting each piece massively improves performance...
# See https://github.com/tensorflow/tensorflow/issues/27692 and
# https://github.com/tensorflow/tensorflow/issues/27692#issuecomment-486619864
def process_pieces(audio, sample_size):
    pieces = []
    idx = 0
    while idx + sample_size <= len(audio):
        piece = audio[idx:idx+sample_size, :]
        pieces.append(np.array(piece))
        idx += sample_size
    return np.array(pieces)

def get_dataset(data_dir, batch_size, sample_rate, sample_size, silence_threshold):
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: load_audio(data_dir, sample_rate, sample_size, silence_threshold),
        output_types=tf.float32,
        output_shapes=((None, 1)), # Not sure about the precise value of this...
    )
    if sample_size:
        dataset = dataset.map(lambda audio: tf.py_function(
            func=process_pieces, inp=[audio, sample_size], Tout=tf.float32
        )).unbatch()
    return dataset.batch(batch_size)