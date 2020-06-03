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
        lambda: load_audio(data_dir, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1)),
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return dataset

# This version (not used) does sub-batching in the dataset itself, rather than in a subsequent iterator.
# N.B: We need to import the quantize function for this.

def sub_batcher(dataset, batch_size, seq_len, overlap, q_type='mu-law', q_levels=256):
    seq_len = seq_len
    q_type = q_type
    q_levels = q_levels
    for batch in dataset:
        num_samps = len(batch[0])
        for i in range(overlap, num_samps, seq_len):
            inputs = quantize(batch[:, i-overlap : i+seq_len], q_type, q_levels) # need to import quantize
            targets = inputs[:, overlap : overlap+seq_len]
            yield (inputs, targets)

def get_dataset_1(data_dir, batch_size, seq_len, overlap):
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: load_audio(data_dir, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return tf.data.Dataset.from_generator(
        lambda: sub_batcher(dataset, batch_size, seq_len, overlap),
        output_types=(tf.int32, tf.int32),
        output_shapes=(
            (batch_size, seq_len + overlap, 1),
            (batch_size, seq_len, 1)
        )
    )