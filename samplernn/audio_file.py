import warnings
import librosa
import soundfile as sf
import numpy as np

# Contains some code adapted from WaveNet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/audio_reader.py

# warnings.simplefilter("always")

def load_audio(files, batch_size):
    '''Generator that yields audio waveforms from the directory.'''
    assert batch_size <= len(files), 'Batch size exceeds the corpus length'
    if not (len(files) % batch_size) == 0:
        warnings.warn('Truncating corpus, length is not equally divisible by batch size')
        files_slice_idx = ( int(np.floor(len(files) / float(batch_size))) * batch_size )
        files = files[:files_slice_idx]
    print('Corpus length: {} files.'.format(len(files)))
    for filename in files:
        (audio, _) = librosa.load(filename, sr=None, mono=True)
        audio = audio.reshape(-1, 1)
        print("Loading corpus entry {}".format(filename))
        yield audio

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)