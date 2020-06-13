# PRiSM SampleRNN  

[PRiSM](https://www.rncm.ac.uk/research/research-centres-rncm/prism/) implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837), for [TensorFlow 2](https://www.tensorflow.org/overview).

-----------
### Table of Contents

* [Features](https://github.com/rncm-prism/prism-samplernn#features)
* [Requirments](https://github.com/rncm-prism/prism-samplernn#requirements)
* [Installation](https://github.com/rncm-prism/prism-samplernn#installation)
* [Architecture](https://github.com/rncm-prism/prism-samplernn#architecture)
* [Training](https://github.com/rncm-prism/prism-samplernn#training)
    - [Preparing Data](https://github.com/rncm-prism/prism-samplernn#preparing-data)
    - [Running the Script](https://github.com/rncm-prism/prism-samplernn#running-the-script)
    - [Statistics](https://github.com/rncm-prism/prism-samplernn#statistics)
    - [Command Line Arguments](https://github.com/rncm-prism/prism-samplernn#command-line-arguments)
    - [Configuring the Model](https://github.com/rncm-prism/prism-samplernn#configuring-the-model)
* [Generating Audio](https://github.com/rncm-prism/prism-samplernn#generating-audio)
* [Resources](https://github.com/rncm-prism/prism-samplernn#resources)
* [Acknowledgements](https://github.com/rncm-prism/prism-samplernn#acknowledgements)
-----------

## Features

- Three-tier architecture
- GRU cell RNN
- Choice of mu-law or linear quantization
- Seeding of generated audio
- Sampling temperature control

-----------

## Requirements

- TensorFlow 2
- Librosa
- Natsort
- Pydub

Note that Pydub is only required for the audio chunking script.

-----------

## Installation
The simplest way to install is with [Anaconda](https://www.anaconda.com/distribution/). After running the installer for your platform, open a new terminal window or tab so that the `conda` package manager is available on your `PATH`. Then create a new environment with:

`conda create -n prism-samplernn anaconda`

We're naming the environment after the repository, but you can choose whatever name you like. Then activate it with:

`conda activate prism-samplernn`

Finally run requirements.txt to install the dependencies:

`pip install -r requirements.txt`

-----------

## Architecture

The architecture of the network conforms to the three-tier design proposed in the original paper, consisting of two upper [RNN](https://www.tensorflow.org/guide/keras/rnn) tiers and a lower [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) tier. The two upper tiers operate on frames of samples, while the lower tier is at the level of individual samples.

-----------

## Training

### Preparing Data

SampleRNN is designed to accept raw audio in the form of .wav files. We therefore need to preprocess our source .wav file by slicing it into chunks, using the supplied [chunk_audio.py](https://bitbucket.org/cmelen/prism-samplernn.py/master/chunk_audio.py) script:
```
python chunk_audio.py <path_to_input_wav> ./chunks/ --chunk_length 8000 --overlap 1000
```
The second argument (required) is the path to the directory to contain the chunks - note the trailing slash (required, otherwise the chunks will be created in the current directory). You will need to create this directory (the above places the chunks in a sub-directory called 'chunks' within the current directory). The script has two optional arguments for setting the chunk_length (defaults to 8000 ms), and an overlap between consecutive chunks (defaults to 0 ms, no overlap).

### Running the Script

Assuming your training corpus is stored in a directory named `data` under the present directory, you can run the train.py script as follows:

```shell
python train.py \
  --id test \
  --data_dir ./data \
  --num_epochs 100 \
  --batch_size 128 \
  --max_checkpoints 2 \
  --checkpoint_every 200 \
  --output_file_dur 3 \
  --sample_rate 16000
```

Temporary checkpoints storing the current state of the model are periodically saved to disk during each epoch, with a permanent checkpoint saved at the end of each epoch. An audio file is also generated at the end of an epoch, which may be used to assess the progress of the training.

### Statistics

Statistics providing information on the progress of traing are printed to the terminal prompt at each step. For example:

`Epoch: 1/2, Step: 125/1500, Loss: 4.182, Accuracy: 13.418, (0.0357 sec/step)`

The `--verbose` command line argument determines how these statistics are printed - if `True` (the default) each step is printed to a new line, if `False` a new line is taken only at each epoch, with each printed step within an epoch being overwritten.

### Command Line Arguments

The following table lists the hyper-parameters that may be passed at the command line:

| Parameter Name             | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `id`                     | Id for the training session          | None           | Yes        |
| `data_dir`               | Path to the directory containing the training data           | None           | Yes        |
| `verbose`                | Set training output verbosity. If `False` training step output is overwritten, if `True` (the default) it is written to a new line.           | None           | No        |
| `logdir_root`            | Location in which to store training log files and checkpoints. All such files are placed in a subdirectory with the id of the training session.           | `./logdir`           | No      |
| `output_dir`             | Path to the directory for audio generated during training           | `./generated`           | No      |
| `config_file`            | File containing the configuration parameters for the training model. Note that this file must contain valid JSON, and have the `.json` extension. | `./default.json`         | No        |
| `num_epochs`             | Number of epochs to run the training | 100           | No        |
| `batch_size`             | Size of the mini-batch. It is recommended that the batch size divide the length of the training corpus without remainder, otherwise the dataset will be truncated to the nearest multiple of the batch size. | 64         | No        |
| `optimizer`              | TensorFlow optimizer to use for training (`adam`, `sgd` or `rmsprop`) | `adam`        | No        |
| `learning_rate`          | Learning rate of the training optimizer   | 0.001         | No        |
| `momentum`               | Momentum of the training optimizer (applies to `sgd` and `rmsprop` only)   | 0.9      | No        |
| `checkpoint_every`       | Interval (in steps) at which to generate a checkpoint file   | 100      | No        |
| `max_checkpoints`        | Maximum number of training checkpoints to keep   | 5      | No        |
| `resume`                 | Whether to resume training from the last available checkpoint   | `True`      | No        |
| `max_generate_per_epoch` | Maximum number of output files to generate at the end of each epoch   | 1      | No        |
| `sample_rate`            | Sample rate of the generated audio | 22050         | No        |
| `output_file_dur`        | Duration of generated audio files (in seconds) | 3         | No        |
| `temperature`            | Sampling temperature for generated audio | 0.95         | No        |
| `seed`                   | Path to audio for seeding when generating audio | None         | No        |
| `seed_offset`            | Starting offset of the seed audio | 0         | No        |
| `val_pcnt`               | Percentage of data to reserve for validation | 0.1         | No        |
| `test_pcnt`              | Percentage of data to reserve for testing | 0.1         | No        |

### Configuring the Model

Model parameters are specified through a JSON configuration file, which may be passed to the training script through the `--config_file` parameter (defaults to ./default.json). The following table lists the available model parameters (note that all parameters are optional and have defaults):

| Parameter Name           | Description           | Default Value  |
| -------------------------|-----------------------|----------------|
| `seq_len`                | RNN sequence length. Note that the value must be evenly-divisible by the top tier frame size.        | 1024           |
| `frame_sizes`            | Frame sizes (in samples) of the two upper tiers in the architecture, in ascending order. Note that the frame size of the upper tier must be an even multiple of that of the lower tier.  | [16,64]            |
| `dim`                    | RNN hidden layer dimensionality          | 1024         | 
| `num_rnn_layers`         | Depth of the RNN in each of the two upper tiers           | 4          |
| `q_type`                 | Quantization type (`mu-law` or `linear`)          | `mu-law`          |
| `q_levels`               | Number of quantization channels (note that if `q_type` is `mu-law` this parameter is ignored, as mu-law quantization requires 256 channels)     | 256           |
| `emb_size`               | Size of the embedding layer in the bottom tier (sample-level MLP)         | 256          |

-----------

## Generating Audio

To generate audio from a trained model use the generate.py script:

```shell
python generate.py \
  --output_path path/to/out.wav \
  --checkpoint_path ./logdir/test/predict/ckpt-100 \
  --config_file ./default.json \
  --num_seqs 10 \
  --dur 10 \
  --sample_rate 22050 \
  --seed path/to/seed.wav \
  --seed_offset 500
```

The model to generate from is specified by the path to an available epoch checkpoint. The generation stage must use the same parameters as the trained model, contained in a JSON config file.

Use the `--num_seqs` parameter to specify the number of audio sequences to generate. Sequences are generated in parallel in a batch, so for large values for `--num_seqs` the operation will be faster than real time.

To seed the generation pass an audio file as the `--seed` parameter. An offset into the file (in samples) may be specified using `--seed_offset`. Note that the size of the seed audio is truncated to the large frame size set during training (64 samples by default).

The following is the full list of command line parameters for generate.py:

| Parameter Name             | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `output_path`              | Path to the generated .wav file          | None           | Yes        |
| `checkpoint_path`          | Path to a saved checkpoint for the model           | None           | Yes        |
| `config_file`              | Path to the JSON config for the model          | None           | Yes        |
| `dur`                      | Duration of generated audio           | 3           | No       |
| `num_seqs`                 | Number of audio sequences to generate          | 1           | No        |
| `sample_rate`              | Sample rate of the generated audio          | 44100           | No        |
| `temperature`              | Sampling temperature for generated audio | 0.95         | No        |
| `seed`                     | Path to audio for seeding when generating audio | None         | No        |
| `seed_offset`              | Starting offset of the seed audio | 0         | No        |

-----------

## Resources

The following is a list of resources providing further information on SampleRNN, and related work:

- [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837) (the original SampleRNN paper)
- [SampleRNN reference implementation](https://github.com/soroushmehr/sampleRNN_ICLR2017) from the authors of the original paper
- [Generating Albums with SampleRNN to Imitate Metal, Rock, and Punk Bands](https://arxiv.org/pdf/1811.06633.pdf) (Dadabots)
- [Generating Music with WaveNet and SampleRNN](https://karlhiner.com/music_generation/wavenet_and_samplernn/)
- [Recurrent Neural Networks (RNN) with TensorFlow / Keras](https://www.tensorflow.org/guide/keras/rnn#build_a_rnn_model_with_nested_inputoutput)

-----------

## Acknowledgements

Thanks are extended to the rest of the [PRiSM team](https://www.rncm.ac.uk/research/research-centres-rncm/prism/prism-team/) for their help and support during development, and especially to [Dr Sam Salem](https://www.rncm.ac.uk/people/sam-salem/) for his immense patience, diligence and perseverance in testing the code.