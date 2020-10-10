# PRiSM SampleRNN  

[PRiSM](https://www.rncm.ac.uk/research/research-centres-rncm/prism/) implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837), for [TensorFlow 2](https://www.tensorflow.org/overview).

-----------

### UPDATES (08/09/20)

* Audio generation speed is now greatly improved (about 3-4 times faster).

### UPDATES (20/08/20)

* Changes to the `chunk_audio.py` script:
   - The `--input_file` and `--output_dir` arguments were previously positional, now they are named arguments.
   - It is no longer necessary to create the output directory beforehand, it will be created it if it does not exist (also no trailing slash is required when supplying the path to the directory).
   - The script output now reports when a chunk is silent and will be omitted.
   - When complete the script now reports the number of chunks processed and number of chunks omitted (for silence).
* A new `--checkpoint_policy` parameter allows to set the policy for saving checkpoints - whether to only save the checkpoint when there has been an improvement in the training metrics (`Best`), or whether to always save them, regardless of changes to the metrics (`Always`).
* Checkpoints for separate training 'runs' under the same id are saved to separate time-stamped directories, with the naming format `DD.MM.YYYY_HH.MM.SS`.
* A new `--resume_from` training script parameter allows a previously saved checkpoint to be passed directly to the script.
* The `--checkpoint_every` parameter now applies at the epoch level only.
* The `--max_checkpoints` parameter, which determines the number of checkpoints retained on disk during training, can now be set to `None`, to retain all checkpoints (no maximum).
* In-training audio generation can be switched off, through the new `--generate` parameter (`True` by default).
* In-training audio generation, when enabled, is aligned with checkpointing frequency (determined by `--checkpoint_every`). Audio is only generated when a new checkpoint has been saved.
* A new `--reduce_learning_rate_after` parameter allows for the learning rate to dynamically adjust itself during training, decaying exponentially after the specified number of epochs.
* A new `--early_stopping_patience` parameter determines the number of epochs without improvement after which training is automatically terminated (defaults to 3).

-----------
### Table of Contents

* [Features](https://github.com/rncm-prism/prism-samplernn#features)
* [Requirments](https://github.com/rncm-prism/prism-samplernn#requirements)
   - [Python Packages](https://github.com/rncm-prism/prism-samplernn#python-packages)
   - [CUDA](https://github.com/rncm-prism/prism-samplernn#cuda)
* [Installation](https://github.com/rncm-prism/prism-samplernn#installation)
* [Architecture](https://github.com/rncm-prism/prism-samplernn#architecture)
* [Training](https://github.com/rncm-prism/prism-samplernn#training)
    - [Preparing Data](https://github.com/rncm-prism/prism-samplernn#preparing-data)
    - [Running the Script](https://github.com/rncm-prism/prism-samplernn#running-the-script)
    - [Statistics](https://github.com/rncm-prism/prism-samplernn#statistics)
    - [Command Line Arguments](https://github.com/rncm-prism/prism-samplernn#command-line-arguments)
    - [Configuring the Model](https://github.com/rncm-prism/prism-samplernn#configuring-the-model)
    - [Resuming Training](https://github.com/rncm-prism/prism-samplernn#resuming-training)
* [Generating Audio](https://github.com/rncm-prism/prism-samplernn#generating-audio)
* [Resources](https://github.com/rncm-prism/prism-samplernn#resources)
* [Acknowledgements](https://github.com/rncm-prism/prism-samplernn#acknowledgements)
-----------

## Features

- Three-tier architecture
- Choice of GRU or LSTM cell RNN
- Choice of mu-law or linear quantization
- Seeding of generated audio
- Sampling temperature control

-----------

## Requirements

### Python Packages

The following Python packages are required:

- TensorFlow 2
- Librosa
- Natsort
- Pydub
- Keras Tuner

Note that Pydub is only required for the audio chunking script, and Keras Tuner for the model tuning script.

### CUDA

Although it is possible to run the training and generation scripts on a CPU alone, optimal (or even tolerable) performance will require a [CUDA-enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus). TensorFlow 2 requires CUDA version 10.1, however installation of CUDA is beyond the scope of this document - for full instructions on how to install CUDA for your platform see the [TensorFlow GPU guide](https://www.tensorflow.org/install/gpu).

-----------

## Installation

If you would prefer to experiment with PRiSM SampleRNN without installing locally we have set up a [Google Colab notebook](https://colab.research.google.com/gist/relativeflux/10573e9e1b10b1ff45e3a00099259741/prism-samplernn.ipynb). Colab is a cloud-based environment for machine learning that provides free access to GPUs. It is an excellent way for those new to machine learning to gain experience with the kinds of tasks and environments involved, with minimal ceremony. Please note that you will need a Google account, and be signed into it, to use the notebook.

To install Prism SampleRNN on a machine to which you have direct access it is **strongly recommended** to use [Anaconda](https://www.anaconda.com/distribution/), a popular open-source platform for scientific computing, which greatly simplifies package management for machine learning projects. After running the installer for your OS, open a new terminal window or tab so that the `conda` package manager is available on your `PATH`. Then create a new environment with:

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

```shell
python chunk_audio.py \
  --input_file path/to/input.wav \
  --output_dir ./chunks \
  --chunk_length 8000 \
  --overlap 1000
```

The `--output_dir` argument specifies the path to the directory to contain the chunks. The directory will be created if it doesn't already exist (the above places the chunks in a sub-directory called 'chunks' within the current directory). The script has two optional arguments for setting the chunk length (defaults to 8000 ms), and an overlap between consecutive chunks (defaults to 0 ms, no overlap).

### Running the Script

Assuming your training corpus is stored in a directory named `data` under the present directory, you can run the train.py script as follows:

```shell
python train.py \
  --id test \
  --data_dir ./data \
  --num_epochs 100 \
  --batch_size 128 \
  --checkpoint_every 5 \
  --output_file_dur 3 \
  --sample_rate 16000
```

Checkpoints storing the current state of the model are periodically saved to disk during training, with the default behaviour being to save a checkpoint at the end of each epoch. This interval may be modified through the `--checkpoint_every` parameter.  By default every checkpoint will be saved, but this behaviour can be controlled using the `--checkpoint_policy` parameter. Pass `Best` to save only the latest best checkpoint according to the training metrics (loss and accuracy). An audio file is also generated each time a checkpoint is saved, which may be used to assess the progress of the training (generation may be switched off by setting `--generate` to `False`).

### Statistics

Statistics providing information on the progress of traing are printed to the terminal prompt at each step. For example:

`Epoch: 1/2, Step: 125/1500, Loss: 4.182, Accuracy: 13.418, (0.0357 sec/step)`

The `--verbose` command line argument determines how these statistics are printed - if `True` (the default) each step is printed to a new line, if `False` a new line is taken only at each epoch, with each printed step within an epoch being overwritten.

### Command Line Arguments

The following table lists the hyper-parameters that may be passed at the command line:

| Parameter Name             | Description           | Default Value  | Required?   |
| ---------------------------|-----------------------|----------------| -----------|
| `id`                     | Id for the training session.          | `default`           | No        |
| `data_dir`               | Path to the directory containing the training data.           | `None`           | Yes        |
| `verbose`                | Set training output verbosity. If `False` training step output is overwritten, if `True` (the default) it is written to a new line.           | `None`           | No        |
| `logdir_root`            | Location in which to store training log files and checkpoints. All such files are placed in a subdirectory with the id of the training session.           | `./logdir`           | No      |
| `output_dir`             | Path to the directory for audio generated during training.           | `./generated`           | No      |
| `config_file`            | File containing the configuration parameters for the training model. Note that this file must contain valid JSON, and should have a name that conforms to the `*.config.json` pattern. | `./default.config.json`         | No        |
| `num_epochs`             | Number of epochs to run the training. | 100           | No        |
| `batch_size`             | Size of the mini-batch. It is recommended that the batch size divide the length of the training corpus without remainder, otherwise the dataset will be truncated to the nearest multiple of the batch size. | 64         | No        |
| `optimizer`              | TensorFlow optimizer to use for training. (`adam`, `sgd` or `rmsprop`) | `adam`        | No        |
| `learning_rate`          | Learning rate of the training optimizer.   | 0.001         | No        |
| `reduce_learning_rate_after`          | Exponentially reduce learning rate after this many epochs.   | `None`         | No        |
| `momentum`               | Momentum of the training optimizer (applies to `sgd` and `rmsprop` only).   | 0.9      | No        |
| `checkpoint_every`       | Interval (in epochs) at which to generate a checkpoint file. Defaults to 1, for every epoch.   | 1      | No        |
| `checkpoint_policy`      | Policy for saving checkpoints - `Always` to save at the epoch interval determined by the value of `checkpoint_every`, or `Best` to save only when the loss and accuracy have improved since the last save.   | `All`      | No        |
| `max_checkpoints`        | Maximum number of checkpoints to keep on disk during training. Defaults to 5. Pass `None` to keep all checkpoints.   | 5      | No        |
| `resume`                 | Whether to resume training, either from the last available checkpoint or from one supplied using the `resume_from` parameter.   | `True`      | No        |
| `resume_from`            | Checkpoint from which to resume training. Ignored when `resume` is `False`.   | `None`      | No        |
| `early_stopping_patience`| Number of epochs with no improvement after which training will be stopped.   | 3      | No        |
| `generate`               | Whether to generate audio output during training. Generation is aligned with checkpoints, meaning that audio is only generated after a new checkpoint has been created.   | `True`      | No        |
| `max_generate_per_epoch` | Maximum number of output files to generate at the end of each epoch.   | 1      | No        |
| `sample_rate`            | Sample rate of the generated audio. | 22050         | No        |
| `output_file_dur`        | Duration of generated audio files (in seconds). | 3         | No        |
| `temperature`            | Sampling temperature for generated audio. Multiple values may be passed, to match the number of sequences to be generated. If the number of values exceeds the value of `--num_seqs`, the list will be truncated to match it. If too few values are provided the last value will be repeated until the list length matches the number of requested sequences. | 0.75         | No        |
| `seed`                   | Path to audio for seeding when generating audio. | `None`         | No        |
| `seed_offset`            | Starting offset of the seed audio. | 0         | No        |
| `num_val_batches`        | Number of batches to reserve for validation. | 1         | No        |

### Configuring the Model

Model parameters are specified through a JSON configuration file, which may be passed to the training script through the `--config_file` parameter. This must have a name which conforms to the `*.config.json` pattern (defaults to `default.config.json`). Note that any configuration file with a name other the name of the supplied default will be ignored by Git (see the `.gitignore` for details). The following table lists the available model parameters (all parameters are optional and have defaults):

| Parameter Name           | Description           | Default Value  |
| -------------------------|-----------------------|----------------|
| `seq_len`                | RNN sequence length. Note that the value must be evenly-divisible by the top tier frame size.        | 1024           |
| `frame_sizes`            | Frame sizes (in samples) of the two upper tiers in the architecture, in ascending order. Note that the frame size of the upper tier must be an even multiple of that of the lower tier.  | [16,64]            |
| `dim`                    | RNN hidden layer dimensionality.          | 1024         |
| `rnn_type`               | RNN type to use, either `gru` or `lstm`.           | `gru`          | 
| `num_rnn_layers`         | Depth of the RNN in each of the two upper tiers.           | 4          |
| `q_type`                 | Quantization type (`mu-law` or `linear`).          | `mu-law`          |
| `q_levels`               | Number of quantization channels (note that if `q_type` is `mu-law` this parameter is ignored, as mu-law quantization requires 256 channels).     | 256           |
| `emb_size`               | Size of the embedding layer in the bottom tier (sample-level MLP).         | 256          |
| `skip_conn`              | Whether to add skip connections to the model's RNN layers.        | `False`          |

### Resuming Training

A training session that has been halted, perhaps by `Ctrl-C`, may be resumed from a previously saved checkpoint. The weights saved to the checkpoint will be loaded into a fresh model, resuming at the last epoch + 1. To enable this set `--resume` to `True`, and optionally the path to a checkpoint through the `--resume_from` parameter (ignored when `--resume` is `False`). If no such checkpoint is supplied the program will search through any previous training run directories for the latest checkpoint. If no checkpoint is found training will begin again from scratch.

-----------

## Generating Audio

To generate audio from a trained model use the generate.py script:

```shell
python generate.py \
  --output_path path/to/out.wav \
  --checkpoint_path ./logdir/default/26.07.2020_20.48.51/model.ckpt-100 \
  --config_file ./default.config.json \
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
| `output_path`              | Path to the generated .wav file.          | `None`           | Yes        |
| `checkpoint_path`          | Path to a saved checkpoint for the model.           | `None`           | Yes        |
| `config_file`              | Path to the JSON config for the model.          | `None`           | Yes        |
| `dur`                      | Duration of generated audio.           | 3           | No       |
| `num_seqs`                 | Number of audio sequences to generate.          | 1           | No        |
| `sample_rate`              | Sample rate of the generated audio.          | 44100           | No        |
| `temperature`              | Sampling temperature for generated audio. Multiple values may be passed, to match the number of sequences to be generated. If the number of values exceeds the value of `--num_seqs`, the list will be truncated to match it. If too few values are provided the last value will be repeated until the list length matches the number of requested sequences.  | 0.75         | No        |
| `seed`                     | Path to audio for seeding when generating audio. | `None`         | No        |
| `seed_offset`              | Starting offset of the seed audio. | 0         | No        |

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