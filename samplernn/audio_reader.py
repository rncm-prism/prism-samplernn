import fnmatch
import os
import random
import re
import threading

import librosa
import sys
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


def load_audio(directory, sample_rate, sample_size, silence_threshold):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print("Corpus length: {} files.".format(len(files)))
    # It does not seem as if os.walk guarantees sort order,
    # so the randomization step is perhaps redundant?
    randomized_files = randomize_files(files)
    i = 0
    for filename in randomized_files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        #audio = preprocess_audio(audio, silence_threshold, sample_size)
        i+=1
        print("Loading corpus entry '{}' ({}/{})".format(filename, i, len(files)))
        #yield audio, filename
        yield audio

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def preprocess_audio(audio, silence_threshold, sample_size):
    audio = copy.deepcopy(audio)
    if silence_threshold is not None:
        # Remove silence
        audio = trim_silence(audio[:, 0], silence_threshold)
        audio = audio.reshape(-1, 1)
        if audio.size == 0:
            print("Warning: {} was ignored as it contains only "
                    "silence. Consider decreasing trim_silence "
                    "threshold, or adjust volume of the audio.")
    pad_elements = sample_size - 1 - \
        (audio.shape[0] + sample_size - 1) % sample_size
    audio = np.concatenate(
        [
            audio,
            np.full(
                (pad_elements, 1),
                0.0,
                dtype='float32'
            )
        ],
        axis=0
    )
    #if sample_size:
    #    while len(audio) >= sample_size:
    #        piece = audio[:sample_size, :]
    #        sess.run(self.enqueue,
    #                    feed_dict={self.sample_placeholder: piece})
    #        audio = audio[sample_size:, :]
    return audio

class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, audio_dir, sample_rate, sample_size=None,
                 silence_threshold=None):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def getDataset(self):
        dataset = tf.data.Dataset.from_generator(
            # See here for why lambda is required: https://stackoverflow.com/a/56602867
            lambda: load_audio(self.audio_dir, self.sample_rate, self.sample_size, self.silence_threshold),
            output_types=tf.float32,
            output_shapes=((None,)), # Not sure about the value of this...
        )
        return dataset
