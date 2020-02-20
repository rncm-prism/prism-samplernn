import fnmatch
import os
import random

import librosa
import soundfile as sf
import copy
import numpy as np
import tensorflow as tf


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
    return files

def load_audio(directory):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print("Corpus length: {} files.".format(len(files)))
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
    #print('Updated wav file at {}'.format(path))