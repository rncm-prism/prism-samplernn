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

# Contains some code adapted from WaveNet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/audio_reader.py

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

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)