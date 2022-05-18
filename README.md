# PRiSM SampleRNN  

[PRiSM](https://www.rncm.ac.uk/research/research-centres-rncm/prism/) implementation of [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837), for [TensorFlow 2](https://www.tensorflow.org/overview).

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
    - [Mixed Precision Training](https://github.com/rncm-prism/prism-samplernn#mixed-precision-training)
    - [Hyperparameter Optimization](https://github.com/rncm-prism/prism-samplernn#hyperparameter-optimization)
        - [Hyperparameter Optimization with Keras Tuner](https://github.com/rncm-prism/prism-samplernn#hyperparameter-optimization-with-keras-tuner)
        - [Hyperparameter Optimization with Ray Tune](https://github.com/rncm-prism/prism-samplernn#hyperparameter-optimization-with-ray-tune)
* [Generating Audio](https://github.com/rncm-prism/prism-samplernn#generating-audio)
* [Resources](https://github.com/rncm-prism/prism-samplernn#resources)
* [Acknowledgements](https://github.com/rncm-prism/prism-samplernn#acknowledgements)
* [Version History](https://github.com/rncm-prism/prism-samplernn#version-history)
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

- [TensorFlow 2](https://www.tensorflow.org/)
- [Librosa](https://librosa.org/doc/latest/index.html)
- [Natsort](https://github.com/SethMMorton/natsort)
- [Pydub](https://github.com/jiaaro/pydub)

Note that Pydub is only required for the audio chunking script.

If you are interested in hyperparameter optimization we provide two scripts especially for that purpose, both of which require the installation of additional Python libraries. For more details on these see the section below on [Hyperparameter Optimization](https://github.com/rncm-prism/prism-samplernn#hyperparameter-optimization).

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

SampleRNN is designed to accept raw audio in the form of .wav files. We therefore need to preprocess our source .wav file by slicing it into chunks, using the supplied [chunk_audio.py](https://github.com/rncm-prism/prism-samplernn/blob/master/chunk_audio.py) script:

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

Checkpoints storing the current state of the model are periodically saved to disk during training, with the default behaviour being to save a checkpoint at the end of each epoch. This interval may be modified through the `--checkpoint_every` parameter.  By default every checkpoint will be saved, but this behaviour can be controlled using the `--checkpoint_policy` parameter. Pass `Best` to save only the latest best checkpoint according to the training metrics (loss and accuracy).

Each time a checkpoint is saved one or more audio files are also generated, which may be used to assess the progress of the training. The number of files generated is controlled by the value of the `--max_generate_per_epoch` parameter. Audio files are saved with names in the format `id_e=1_t=0.95`, with `id` being the id of the current training, `e` the current epoch and `t` the temperature at which the audio was generated. In-training generation may be switched off by setting `--generate` to `False`.

Before training begins a certain portion of the input dataset is reserved for _validation_, which occurs at the end of each epoch. This is a stage during which the network is exposed to (but not trained on) a small portion of the dataset, the purpose of which is to test whether the network can generalize to inputs that it has not seen before. How well the network does when processing this unseen data can provide insight into whether the network is [overfitting or underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) (see the [Statistics](https://github.com/rncm-prism/prism-samplernn#statistics) section below for how validation statistics are reported). The size of the validation set is determined by the `val_frac` parameter, which defaults to 0.1, yielding a 9/1 split between training and validation data, the most common division (note that the actual values are rounded to the nearest multiple of the batch size).

### Statistics

Statistics providing information on the progress of traing are printed to the terminal prompt at each step. For example:

`Epoch: 1/2, Step: 125/1500, Loss: 4.182, Accuracy: 13.418, (0.0357 sec/step)`

At the end of each epoch the final loss and accuracy for the epoch are displayed, along with the validation loss and validation accuracy, and the epoch's total duration:

`Epoch: 1/2, Loss: 2.342, Accuracy: 18.928, Val Loss: 2.194, Val Accuracy: 19.273 (1500 steps in 1 min 11.235 sec)`

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
| `device`                 | GPU allocation. Set this to the number of a specific device (indexed from 0), or pass `All` (the default) to allow all visible devices to be used.          | `All`           | No        |
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
| `sample_rate`            | Sample rate of the generated audio. | 16000         | No        |
| `output_file_dur`        | Duration of generated audio files (in seconds). | 3         | No        |
| `temperature`            | Sampling temperature for generated audio. Multiple values may be passed, to match the number of sequences to be generated. If the number of values exceeds the value of `--num_seqs`, the list will be truncated to match it. If too few values are provided the last value will be repeated until the list length matches the number of requested sequences. | 0.95         | No        |
| `seed`                   | Path to audio for seeding when generating audio. | `None`         | No        |
| `seed_offset`            | Starting offset of the seed audio. | 0         | No        |
| `val_frac`               | Fraction of the dataset to be set aside for validation, rounded to the nearest multiple of the batch size. Defaults to 0.1, or 10%. | 0.1         | No        |
| `mixed_precision`               | Whether to run the training in mixed precision mode, which sets the floating point precision of some internal layers of the model to 16-bits. This can greatly speed up training step time. | `False`         | No        |

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

### Mixed Precision Training

A significant speed-up in training time can be achieved by running the training in mixed precision mode, by setting the value of the `--mixed_precision` hyperparameter to `True` (the default is `False`). Mixed precision is a mode of the model which involves the use of both 16-bit and 32-bit floating-point types. This means that less memory is used when training the model, which can greatly speed up training time. In fact in our own experiments we have found an increase in training step speed of around 3.5 times (although YMMV).

Generation using mixed precision has not yet been implemented, but the current generate.py script can be used to generate from models trained in either mode.

For more on the underlying TensorFlow mixed precision implementation see the [official documentation](https://www.tensorflow.org/guide/mixed_precision).

Note that mixed precision training will only show a significant increase in training speed with recent NVIDIA GPUs, with compute capability 7.0 or higher. In particular GPUs provided by Google Colab will typically be from much older generations, which will not benefit from mixed precision training.

### Hyperparameter Optimization

The variables which control the training process are known as _hyperparameters_. Typically these will remain fixed over the course of a single training session, as opposed to the model's internal parameters - its weights and biases - which are updated at each step. Hyperparameters determine the model's overall performance, so it is important to pick the right ones. This is often a difficult problem, but fortunately it is possible to automate the process. We have included two scripts, `keras_tuner.py` and `ray_tune.py`, which each use a separate hyperparameter tuning systems. Both work by defining a hyperparameter 'search space', which the system can examine to find the optimal set of hyperparameters, a process know as hyperparameter tuning or optimization. Details for how to use each script can be found below.

#### Hyperparameter Optimization with Keras Tuner

[Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) is a library for automated hyperparameter tuning from the developers of [Keras](https://keras.io/). Install the latest version (inside the conda environment) with:

`pip install -U keras-tuner`

The `keras_tuner.py` script is very similar to `train.py`, except that instead of taking just a single value for each hyperparameter it takes a list of values, defining the search space for that hyperparameter. No separate JSON configuration file is required for the model, all hyperparameters being passed through the command line arguments. To run the script, execute:

```shell
python keras_tuner.py \
  --data_dir path/to/dataset \
  --num_epochs 20 \
  --frame_sizes 16 64 \
  --frame_sizes 32 128 \
  --batch_size 32 64 128 \
  --seq_len 512 1024 \
  --num_rnn_layers 2 4
```

Note that the frame sizes, which determine the number of samples consumed in a single timestep by each RNN tier, are here specified by two separate entries of the `frame_sizes` argument. Each defines a separate entry in the search space.

For the full list of parameters to `keras_tuner.py` run the script with the `--help` option.

At the end of the tuning process the script will print a summary of the best trial to standard output, in the following format (`Score` is simply the objective, which defaults to the `val_loss`):

```shell
[Trial summary]
 |-Trial ID: b366a8c644b6b0a28ef0c378886c81fe
 |-Score: 0.9532838344573975
 |-Best step: 3
 > Hyperparameters:
 |-batch_size: 64
 |-frame_sizes: 32 128
 |-dim: 1024
 |-learning_rate: 0.001
 |-momentum: 0.95
 |-num_rnn_layers: 2
 |-rnn_dropout: 0.25
 |-seq_len: 512
```

For more information on Keras Tuner see its [documentation pages](https://keras-team.github.io/keras-tuner/).

#### Hyperparameter Optimization with Ray Tune

[Ray Tune](https://docs.ray.io/en/master/tune/index.html) is a library found within [Ray](https://ray.io/), a framework for distributed computing with Python. Unlike `keras_tuner.py`, the `ray_tune.py` script allows for tuning across multiple GPUs simultaneously, greatly increasing the speed and efficiency of the tuning process. Install the latest version of Ray (inside the conda environment) with:

`pip install -U ray`

You might also need to install a few additional packages:

`pip install tensorboardX`

`pip install 'ray[tune]'`

The `ray_tune.py` script closely resembles `keras_tuner.py`, except for the addition of a few extra parameters:

```shell
python ray_tune.py \
  --data_dir path/to/dataset \
  --num_cpus 4 \
  --num_gpus 2 \
  --num_trials 10 \
  --num_epochs 20 \
  --frame_sizes 16 64 \
  --frame_sizes 32 128 \
  --batch_size 32 64 128 \
  --seq_len 512 1024 \
  --num_rnn_layers 2 4
```

As with `keras_tuner.py`, the frame sizes are specified by two separate entries of the `frame_sizes` argument. The additional arguments relate to distributed training - `--num_cpus`, which specifies the number of cpu cores to allocate, and `--num_gpus`, which specifies the number of gpus to use (which can be a single gpu).

For the full list of parameters to `ray_tune.py` run the script with the `--help` option.

The final output, when all trials have been run, is a JSON object containing the best hyperparameters, for example:

```shell
{
  "batch_size": 128,
  "dim": 2048,
  "frame_sizes": [
    32,
    128
  ],
  "learning_rate": 0.001,
  "momentum": 0.5,
  "num_rnn_layers": 2,
  "q_type": "mu-law",
  "rnn_dropout": 0.6,
  "rnn_type": "gru",
  "seq_len": 1024,
  "skip_conn": true
}
```

A summary of the best trial, including its metadata (such as training iterations, total duration, etc.), will also be printed.

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
  --sample_rate 16000 \
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
| `device`                   | GPU allocation. Set this to the number of a specific device (indexed from 0), or pass `All` (the default) to allow all visible devices to be used.          | `All`           | No        |
| `dur`                      | Duration of generated audio.           | 3           | No       |
| `num_seqs`                 | Number of audio sequences to generate.          | 1           | No        |
| `sample_rate`              | Sample rate of the generated audio.          | 16000           | No        |
| `temperature`              | Sampling temperature for generated audio. Multiple values may be passed, to match the number of sequences to be generated. If the number of values exceeds the value of `--num_seqs`, the list will be truncated to match it. If too few values are provided the last value will be repeated until the list length matches the number of requested sequences.  | 0.95         | No        |
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

Thanks also to [Tanguy Pocquet](https://linkedin.com/in/tanguy-pocquet-2bb72b1ba/) for his contribution to developing and testing the mixed precision training code, and also [Dr Bofan Ma](https://mabofan.com) for testing the scripts on the MacBook M1 chip.

-----------

## Version History

## 26/03/22

* Training with mixed precision floating point types is now available, using the `--mixed_precision` command line parameter to train.py.
* It is also now possible to target a specific GPU for training or for generation, with the new `--device` parameter. On machines with multiple GPUs this means that multiple training sessions can be run simultaneously.

## 24/11/21

* Fixed Windows-specific bug identified in [issue#26](https://github.com/rncm-prism/prism-samplernn/issues/26#issue-1037458893), caused by hardcoded posix pathname in generate.py.

## 10/08/21

* Fixed bug which occurred if not passing an explicit value for the temperature parameter, either in train.py or generate.py - the internal default was not being forwarded in the correct format.

## 07/08/21

* Generated audio files are now saved with names in the format `id_e=1_t=0.95`, with `id` being the id of the current training, `e` the current epoch and `t` the temperature at which the audio was generated.
* The default sampling temperature for generation is now 0.95.
* The default sample rate is now 16kHz.

### 14/03/21

* Added new hparam optimization module, [ray_tune.py](https://github.com/rncm-prism/prism-samplernn/blob/master/ray_tune.py), based on [Ray Tune](https://docs.ray.io/en/master/tune/index.html).
* Renamed the old tune.py module to [keras_tuner.py](https://github.com/rncm-prism/prism-samplernn/blob/master/keras_tuner.py), since it uses [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner).
* The size of the validation set is now computed as a fraction of the total dataset size. This is configured using the new `val_frac` parameter, which defaults to 0.1 (= 10% of total dataset). The old `num_val_batches` is retained for now, but has no function and will be removed in a future release.

### 03/12/20

* Implemented validation step.
* Added tuner script for hyperparameter optimization.
* Removed ReLU activation from the final MLP layer.
* Fixed linear quantization bug.

### 08/09/20

* Audio generation speed is now greatly improved (about 3-4 times faster).

### 20/08/20

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