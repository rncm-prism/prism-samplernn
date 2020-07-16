from __future__ import print_function
import argparse
import os
import time
import json
from platform import system

import tensorflow as tf
import numpy as np
import librosa

from samplernn import (SampleRNN, find_files, quantize)
from dataset import get_dataset
from generate import generate


# https://github.com/ibab/tensorflow-wavenet/issues/255
LOGDIR_ROOT = 'logdir' if system()=='Windows' else './logdir'
OUTDIR = './generated'
CONFIG_FILE = './default.config.json'
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SILENCE_THRESHOLD = None
OUTPUT_DUR = 3 # Duration of generated audio in seconds
CHECKPOINT_EVERY = 200
MAX_CHECKPOINTS = 5
SAMPLE_RATE = 22050 # Sample rate of generated audio
SAMPLING_TEMPERATURE = 0.75
SEED_OFFSET = 0
MAX_GENERATE_PER_EPOCH = 1
RESUME = True
VAL_PCNT = 0.1
TEST_PCNT = 0.1


def get_arguments():

    def check_bool(value):
        val = str(value).upper()
        if 'TRUE'.startswith(val):
            return True
        elif 'FALSE'.startswith(val):
            return False
        else:
           raise ValueError('Argument is neither `True` nor `False`')

    def check_positive(value):
        val = int(value)
        if val < 1:
             raise argparse.ArgumentTypeError("%s is not positive" % value)
        return val

    parser = argparse.ArgumentParser(description='PRiSM TensorFlow SampleRNN')
    parser.add_argument('--data_dir',                   type=str,            required=True,
                                                        help='Path to the directory containing the training data')
    parser.add_argument('--id',                         type=str,            default='default', help='Id for the current training session')
    parser.add_argument('--verbose',                    type=check_bool,
                                                        help='Whether to print training step output to a new line each time (the default), or overwrite the last output', default=True)
    parser.add_argument('--batch_size',                 type=check_positive, default=BATCH_SIZE, help='Size of the mini-batch')
    parser.add_argument('--logdir_root',                type=str,            default=LOGDIR_ROOT,
                                                        help='Root directory for training log files')
    parser.add_argument('--config_file',                type=str,            default=CONFIG_FILE,
                                                        help='Path to the JSON config for the model')
    parser.add_argument('--output_dir',                 type=str,            default=OUTDIR,
                                                        help='Path to the directory for audio generated during training')
    parser.add_argument('--output_file_dur',            type=check_positive, default=OUTPUT_DUR,
                                                        help='Duration of generated audio files (in seconds)')
    parser.add_argument('--sample_rate',                type=check_positive, default=SAMPLE_RATE,
                                                        help='Sample rate of the generated audio')
    parser.add_argument('--num_epochs',                 type=check_positive, default=NUM_EPOCHS,
                                                        help='Number of training epochs')
    parser.add_argument('--checkpoint_every',           type=check_positive, default=CHECKPOINT_EVERY,
                                                        help='Interval (in steps) at which to generate a checkpoint file')
    parser.add_argument('--optimizer',                  type=str,            default='adam', choices=optimizer_factory.keys(),
                                                        help='Type of training optimizer to use')
    parser.add_argument('--learning_rate',              type=float,          default=LEARNING_RATE,
                                                        help='Learning rate of training')
    parser.add_argument('--momentum',                   type=float,          default=MOMENTUM,
                                                        help='Optimizer momentum')
    #parser.add_argument('--silence_threshold',          type=float,          default=SILENCE_THRESHOLD)
    parser.add_argument('--max_checkpoints',            type=check_positive, default=MAX_CHECKPOINTS,
                                                        help='Maximum number of training checkpoints to keep')
    parser.add_argument('--resume',                     type=check_bool,     default=RESUME,
                                                        help='Whether to resume training from the last available checkpoint')
    parser.add_argument('--max_generate_per_epoch',     type=check_positive, default=MAX_GENERATE_PER_EPOCH,
                                                        help='Maximum number of output files to generate at the end of each epoch')
    parser.add_argument('--temperature',                type=float,          default=SAMPLING_TEMPERATURE,
                                                        help='Sampling temperature for generated audio')
    parser.add_argument('--seed',                       type=str,            help='Path to audio for seeding')
    parser.add_argument('--seed_offset',                type=int,            default=SEED_OFFSET,
                                                        help='Starting offset of the seed audio')
    parser.add_argument('--val_pcnt',                   type=float,          default=VAL_PCNT,
                                                        help='Percentage of data to reserve for validation')
    parser.add_argument('--test_pcnt',                  type=float,          default=TEST_PCNT,
                                                        help='Percentage of data to reserve for testing')
    return parser.parse_args()

# Optimizer factory adapted from WaveNet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py

def create_adam_optimizer(learning_rate, momentum):
    return tf.optimizers.Adam(learning_rate=learning_rate,
                              epsilon=1e-4)

def create_sgd_optimizer(learning_rate, momentum):
    return tf.optimizers.SGD(learning_rate=learning_rate,
                             momentum=momentum)

def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.optimizers.RMSprop(learning_rate=learning_rate,
                                 momentum=momentum,
                                 epsilon=1e-5)

optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def create_model(batch_size, config):
    seq_len = config.get('seq_len')
    frame_sizes = config.get('frame_sizes')
    q_type = config.get('q_type')
    q_levels = 256 if q_type=='mu-law' else config.get('q_levels')
    assert frame_sizes[0] < frame_sizes[1], 'Frame sizes should be specified in ascending order'
    # The following model configuration interdependencies are sourced from the original implementation:
    # https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/models/three_tier/three_tier.py
    assert seq_len % frame_sizes[1] == 0, 'seq_len should be evenly divisible by tier 2 frame size'
    assert frame_sizes[1] % frame_sizes[0] == 0, 'Tier 2 frame size should be evenly divisible by tier 1 frame size'
    return SampleRNN(
        batch_size=batch_size,
        frame_sizes=frame_sizes,
        seq_len=seq_len,
        q_type=q_type,
        q_levels=q_levels,
        dim=config.get('dim'),
        rnn_type=config.get('rnn_type'),
        num_rnn_layers=config.get('num_rnn_layers'),
        emb_size=config.get('emb_size'),
        skip_conn=config.get('skip_conn')
    )


def main():
    args = get_arguments()

    # Create training session directories
    if not find_files(args.data_dir):
        raise ValueError("No audio files found in '{}'.".format(args.data_dir))
    logdir = os.path.join(args.logdir_root, args.id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logdir_train = os.path.join(logdir, 'train')
    if not os.path.exists(logdir_train):
        os.makedirs(logdir_train)
    logdir_predict = os.path.join(logdir, 'predict')
    if not os.path.exists(logdir_predict):
        os.makedirs(logdir_predict)
    generate_dir = os.path.join(args.output_dir, args.id)
    if not os.path.exists(generate_dir):
        os.makedirs(generate_dir)

    # Load model configuration
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)
    model = create_model(args.batch_size, config)

    seq_len = model.seq_len
    q_type = model.q_type
    q_levels = model.q_levels

    # Optimizer
    opt = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    model.compile(optimizer=opt, loss=compute_loss, metrics=[train_accuracy])

    overlap = model.big_frame_size
    dataset = get_dataset(args.data_dir, args.batch_size, model.seq_len, overlap)

    # Dataset iterator
    def train_iter():
        for batch in dataset:
            num_samps = len(batch[0])
            for i in range(overlap, num_samps, seq_len):
                x = quantize(batch[:, i-overlap : i+seq_len], q_type, q_levels)
                y = x[:, overlap : overlap+seq_len]
                yield (x, y)

    def get_steps_per_epoch():
        files = find_files(args.data_dir)
        num_batches = len(files) // args.batch_size
        (samples, _) = librosa.load(files[0], sr=None, mono=True)
        steps_per_batch = int(np.floor(len(samples) / float(model.seq_len)))
        return num_batches * steps_per_batch

    training_start_time = time.time()

    steps_per_epoch = get_steps_per_epoch()

    init_data = np.random.randint(0, model.q_levels, (model.batch_size, overlap + model.seq_len, 1))
    model(init_data)
    model.reset_states()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='{0}/ckpt_{{epoch}}'.format(logdir_train),
            save_freq=args.checkpoint_every),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='{0}/model.ckpt_{{epoch}}'.format(logdir_predict),
            save_weights_only=True,
            save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            update_freq=50)
    ]

    # Train
    verbose = 1 if args.verbose==True else 2
    try:
        model.fit(
            train_iter(),
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            shuffle=False,
            callbacks=callbacks,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        print()
        print('Keyboard interrupt')
        print()


if __name__ == '__main__':
    main()
