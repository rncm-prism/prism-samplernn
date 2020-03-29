from __future__ import division
import tensorflow as tf



def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.math.log(1 + mu * safe_audio_abs) / tf.math.log(1. + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def linear_quantize(samples, q_levels):
    '''Floats in (-1, 1) to ints in [0, q_levels-1]'''
    epsilon = 1e-5
    out = samples.numpy().copy()
    out -= out.min(axis=1)[:, None]
    out *= ((q_levels - epsilon) / out.max(axis=1))[:, None]
    out += epsilon / 2
    return out.astype('int32')

def linear_quantize2(samples, q_levels):
    epsilon = 1e-2
    out = samples.numpy().copy()
    out -= out.min(axis=1)[:, None]
    out /= out.max(axis=1)[:, None]
    out *= q_levels - epsilon
    out += epsilon / 2
    return out.astype('int32')

def linear_dequantize(samples, q_levels):
    return tf.cast(samples, tf.float32) / (q_levels / 2) - 1

def quantize(data, type='mu-law', q_levels=256):
    if type=='mu-law':
        return mu_law_encode(data, 256)
    elif type=='linear':
        return linear_quantize(data, q_levels)

def dequantize(data, type='mu-law', q_levels=256):
    if type=='mu-law':
        return mu_law_decode(data, 256)
    elif type=='linear':
        return linear_dequantize(data, q_levels)

def one_hot_encode(input, batch_size, q_levels):
    '''One-hot encodes the waveform amplitudes.

    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    encoded = tf.one_hot(
        input,
        depth=q_levels,
        dtype=tf.float32,
    )
    shape = [batch_size, -1, q_levels]
    return tf.reshape(encoded, shape)

# https://discuss.pytorch.org/t/equivalent-to-torch-unsqueeze-in-tensorflow/26379
def unsqueeze(input, axis=0):
    return tf.expand_dims(input, axis)
