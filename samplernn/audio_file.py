import warnings
import librosa
import soundfile as sf
import numpy as np
import random

# Contains some code adapted from WaveNet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/audio_reader.py

# warnings.simplefilter("always")

def randomize(list):
    list_idx = [i for i in range(len(list))]
    random.shuffle(list_idx)
    for idx in range(len(list)):
        yield list[list_idx[idx]]

def yield_from_list(list, shuffle=True):
    list_idx = [i for i in range(len(list))]
    if shuffle==True : random.shuffle(list_idx)
    for idx in range(len(list)):
        yield list[list_idx[idx]]

def load_audio(files, shuffle=True):
    '''Generator that yields audio waveforms from the directory.'''
    print('Corpus length: {} files.'.format(len(files)))
    for filename in yield_from_list(files, shuffle=shuffle):
        (audio, _) = librosa.load(filename, sr=None, mono=True)
        audio = audio.reshape(-1, 1)
        print("Loading corpus entry {}".format(filename))
        yield audio

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)