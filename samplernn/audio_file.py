import fnmatch
import os
import random
import warnings

import librosa
import soundfile as sf
from natsort import natsorted
import copy
import numpy as np
import tensorflow as tf


# warnings.simplefilter("always")

def randomize_files(files):
    files_idx = [i for i in range(len(files))]
    random.shuffle(files_idx)
    for idx in range(len(files)):
        yield files[files_idx[idx]]

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return natsorted(files)

def load_audio(directory, batch_size):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    assert batch_size <= len(files), 'Batch size exceeds the corpus length'
    if not (len(files) % batch_size) == 0:
        warnings.warn('Truncating corpus, length is not equally divisible by batch size')
        files_slice_idx = ( int(np.floor(len(files) / float(batch_size))) * batch_size )
        files = files[:files_slice_idx]
    print('Corpus length: {} files.'.format(len(files)))
    # It does not seem as if os.walk guarantees sort order,
    # so the randomization step is perhaps redundant?
    for filename in randomize_files(files):
        (audio, _) = librosa.load(filename, sr=None, mono=True)
        audio = audio.reshape(-1, 1)
        print("Loading corpus entry {}".format(filename))
        yield audio

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)