# PRiSM SampleRNN  

PRiSM implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837), using TensorFlow 2.

## Features

- Three-tier architecture
- GRU cell RNN
- Mu-law quantization

## Requirements
- TensorFlow 2
- Librosa

## Installation
The simplest way to install is with [Anaconda](https://www.anaconda.com/distribution/). When you have that all set up create a new environment with:

`conda create -n prism-samplernn anaconda`

We're naming the environment after the repo, but you can choose whatever name you like. Then activate it with:

`conda activate prism-samplernn`

Then run requirements.txt to install TensorFlow and Librosa:

`pip install -r requirements.txt`

## Training

### Preparing data

SampleRNN is designed to accept raw audio in the form of .wav files. We therefore need to preprocess our source .wav file by slicing it into chunks, using the supplied [chunk_audio.py](https://bitbucket.org/cmelen/prism-samplernn.py/master/chunk_audio.py) script:
```
python chunk_audio.py <path_to_input_wav> ./chunks/ --chunk_length 8000 --overlap 4000
```
The second argument (required) is the path to the directory to contain the chunks - note the trailing slash (required, otherwise the chunks will be created in the current directory). You will need to create this directory (the above places the chunks in a sub-directory called 'chunks' within the current directory). The script has two optional arguments for setting the chunk_length (defaults to 8000 ms), and the overlap betweem consecutive chunks (defaults to 4000 ms).

### Running the training script

Assuming your training corpus is stored in a directory named `data` under the present directory, you can run the train.py script as follows:

```shell
python train.py \
	--data_dir ./data \
	--num_epochs 10 \
	--frame_sizes 2 8 \
	--batch_size 1 \
	--seq_len 1024 \
	--optimizer adam
```

Type `python train.py --help` for a list of all available parameters and their default values.

