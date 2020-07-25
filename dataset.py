import numpy as np
import tensorflow as tf
from samplernn import load_audio


def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = ( int(np.floor(len(batch[0]) / float(seq_len))) * seq_len )
    zeros = np.zeros([batch_size, overlap, 1], dtype='float32')
    return tf.concat([zeros, batch[:, :num_samps, :]], axis=1)

def get_dataset(data_dir, num_epochs, batch_size, seq_len, overlap):
    dataset = tf.data.Dataset.from_generator(
        lambda: load_audio(data_dir, batch_size),
        output_types=tf.float32,
        output_shapes=((None, 1)),
    )
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    #return dataset.repeat(num_epochs)
    return dataset
