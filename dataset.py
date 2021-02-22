import fnmatch
import os
import random
import numpy as np
import tensorflow as tf
from samplernn import (load_audio, quantize)


def round_to(x, base=5):
    return base * round(x/base)

def truncate_to(x, base):
    return int(np.floor(x / float(base))) * base

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

# We need an initial random shuffle which remains the same if we resume, we could use the second
# argument to random.shuffle, but that's a BAD idea, see https://stackoverflow.com/a/19307329/795131
# and https://stackoverflow.com/a/29684037/795131. The right way is to instantiate our own
# random.Random instance, to get both decent randomness AND avoid polluting the global environment.
def get_dataset_filenames_split(data_dir, val_frac, batch_size):
    files = find_files(data_dir)
    assert batch_size <= len(files), 'Batch size exceeds the corpus length'
    if not files:
        raise ValueError(f'No wav files found in {data_dir}.')
    random.Random(4).shuffle(files)
    # Truncate to the closest batch_size multiple.
    if not (len(files) % batch_size) == 0:
        warnings.warn('Truncating dataset, length is not equally divisible by batch size')
        idx = truncate_to(len(files), batch_size)
        files = files[: idx]
    val_size = len(files) * val_frac
    val_size = round_to(val_size, batch_size)
    if val_size == 0 : val_size = batch_size
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
        lambda: load_audio(files, shuffle=shuffle),
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
