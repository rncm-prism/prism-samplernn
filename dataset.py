import numpy as np
import tensorflow as tf
from samplernn import load_audio


def pad_batch_old(batch, batch_size, seq_len, overlap):
    num_samps = len(batch[0])
    padding = ( seq_len - 1 - (num_samps + seq_len - 1) % seq_len ) + overlap
    padded_batch = np.zeros([batch_size, num_samps + padding, 1], dtype='float32')
    for (i, samples) in enumerate(batch):
        padded_batch[i, overlap : overlap + len(samples), :] = samples
    return padded_batch

def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def get_dataset(data_dir, batch_size, seq_len, overlap):
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: load_audio(data_dir, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1)),
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return dataset

