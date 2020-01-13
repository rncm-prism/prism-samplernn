# PRiSM SampleRNN  

PRiSM implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837). Uses Tensorflow 1.2.

Forked from https://github.com/gogobd/SampleRNN. Implements a 3-tier SampleRNN architecture, with mu-law quantization.

## Requirements
- Tensroflow 1.2  
- Python 3.6  
- Librosa   

## Installation
To install on you will need Anaconda. When you have that all set up open the Anaconda prompt and create a new environment with:

`conda create -n prism-samplernn python=3.6 anaconda`

Our Anaconda environment uses Python 3.6. We're naming the environment after the repo, but you can choose whatever name you like. Then activate it with:

`conda activate prism-samplernn`

Then run requirements.txt to install Tensorflow and Librosa:

`pip install -r requirements.txt`

## Input Data

SampleRNN is designed to accept raw audio in the form of .wav files. We therefore need to preprocess our source .wav file by slicing it into chunks, using the supplied [chunk_audio.py](https://bitbucket.org/cmelen/prism-samplernn.py/master/chunk_audio.py) script:
```
python chunk_audio.py <path_to_input_wav> ./chunks/ --chunk_length 8000 --overlap 4000
```
The second argument (required) is the path to the directory to contain the chunks - note the trailing slash (required, otherwise the chunks will be created in the current directory). You will need to create this directory (the above places the chunks in a sub-directory called 'chunks' within the current directory). The script has two optional arguments for setting the chunk_length (default to 8000 ms), and the overlap betweem consecutive chunks (defaults to 4000 ms).

## Training 
```shell
python train.py \
	--data_dir=./pinao-corpus \
	--silence_threshold=0.1 \
	--sample_size=102408 \
	--big_frame_size=8 \
	--frame_size=2 \
	--q_levels=256 \
	--rnn_type=GRU \
	--dim=1024 \
	--n_rnn=1 \
	--seq_len=520 \
	--emb_size=256 \
	--batch_size=64 \
	--optimizer=adam \
	--num_gpus=4
```
or  
```shell
sh run.sh
```
## Related projects
This work is based on the flowing implementations with some modifications:  
* [tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet), a TensorFlow implementation of WaveNet
* [sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017), a Theano implementation of sampleRNN
