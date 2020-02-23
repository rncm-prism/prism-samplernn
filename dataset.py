import numpy as np
import tensorflow as tf
from samplernn import load_audio


def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = len(batch[0])
    padding = ( seq_len - 1 - (num_samps + seq_len - 1) % seq_len ) + overlap
    padded_batch = np.zeros([batch_size, num_samps + padding, 1], dtype='float32')
    for (i, samples) in enumerate(batch):
        padded_batch[i, :len(samples), :] = samples
    return padded_batch

def get_dataset(data_dir, num_epochs, batch_size, seq_len, overlap):
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: load_audio(data_dir),
        output_types=tf.float32,
        output_shapes=((None, 1)), # Not sure about the precise value of this...
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return dataset

