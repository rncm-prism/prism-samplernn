import fnmatch
import os
import random
import numpy as np
import tensorflow as tf
from samplernn import (load_audio, quantize)


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def get_dataset_filenames_split(data_dir, val_size):
    files = find_files(data_dir)
    if not files:
        raise ValueError("No audio files found in '{}'.".format(data_dir))
    random.shuffle(files)
    val_start = len(files) - val_size
    return files[: val_start], files[val_start :]

def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def get_subseq(dataset, batch_size, seq_len, overlap, q_type, q_levels):
    for batch in dataset:
        batch = quantize(batch, q_type, q_levels)
        num_samps = len(batch[0])
        for i in range(overlap, num_samps, seq_len):
            x = batch[:, i-overlap : i+seq_len]
            y = x[:, overlap : overlap+seq_len]
            yield (x, y)

def get_dataset(files, num_epochs, batch_size, seq_len, overlap, drop_remainder=False, shuffle=True, q_type='mu-law', q_levels=256):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(files, batch_size, shuffle=shuffle),
        output_types=tf.float32,
        output_shapes=((None, 1))
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size, drop_remainder)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return tf.data.Dataset.from_generator(
        lambda: get_subseq(dataset, batch_size, seq_len, overlap, q_type, q_levels),
        output_types=(tf.int32, tf.int32),
        output_shapes=(
            (batch_size, seq_len + overlap, 1),
            (batch_size, seq_len, 1)))
