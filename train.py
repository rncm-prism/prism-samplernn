from __future__ import print_function
import argparse
import os
import time
from datetime import datetime
import json
from platform import system
import logging

# Disable verbose TensorFlow looging...
# See https://github.com/LucaCappelletti94/silence_tensorflow
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

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
CHECKPOINT_POLICY = 'All' # 'All' or 'Best'
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
                                                        help='Interval (in epochs) at which to generate a checkpoint file')
    parser.add_argument('--optimizer',                  type=str,            default='adam', choices=optimizer_factory.keys(),
                                                        help='Type of training optimizer to use')
    parser.add_argument('--learning_rate',              type=float,          default=LEARNING_RATE,
                                                        help='Learning rate of training')
    parser.add_argument('--reduce_learning_rate_after', type=int,            default=0,
                                                        help='Exponentially reduce learning rate after this many epochs')
    parser.add_argument('--momentum',                   type=float,          default=MOMENTUM,
                                                        help='Optimizer momentum')
    #parser.add_argument('--silence_threshold',          type=float,          default=SILENCE_THRESHOLD)
    parser.add_argument('--checkpoint_policy',          type=str, default=CHECKPOINT_POLICY, choices=['All', 'Best'],
                                                        help='Policy for saving checkpoints')
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

def get_latest_checkpoint(logdir):
    rundirs = os.listdir(logdir)
    if len(rundirs) > 0:
        rundir_datetimes = []
        for f in rundirs:
            if os.path.isdir(os.path.join(logdir, f)):
                dt = datetime.strptime(f, '%d.%m.%Y_%H.%M.%S')
                rundir_datetimes.append(dt)
        latest_rundir = max(rundir_datetimes).strftime('%d.%m.%Y_%H.%M.%S')
        return tf.train.latest_checkpoint(os.path.join(logdir, latest_rundir))

def main():
    args = get_arguments()

    files = find_files(args.data_dir)
    if not files:
        raise ValueError("No audio files found in '{}'.".format(args.data_dir))
    dataset_size = len(files)

    # Create training session directories
    logdir = os.path.join(args.logdir_root, args.id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    generate_dir = os.path.join(args.output_dir, args.id)
    if not os.path.exists(generate_dir):
        os.makedirs(generate_dir)
    rundir = '{}/{}'.format(logdir, datetime.now().strftime('%d.%m.%Y_%H.%M.%S'))

    # Load model configuration
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)
    model = create_model(args.batch_size, config)

    seq_len = model.seq_len
    overlap = model.big_frame_size
    q_type = model.q_type
    q_levels = model.q_levels

    # Optimizer
    opt = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=opt, loss=compute_loss, metrics=[train_accuracy])

    def get_initial_epoch():
        latest_checkpoint = get_latest_checkpoint(logdir)
        if latest_checkpoint:
            epoch = int(latest_checkpoint.split('/')[-1].split('-')[-1])
        else:
            epoch = 0
        return epoch

    initial_epoch = get_initial_epoch()
    dataset = get_dataset(args.data_dir, args.num_epochs - initial_epoch, args.batch_size, seq_len, overlap)

    # Dataset iterator
    def train_iter():
        for batch in dataset:
            num_samps = len(batch[0])
            for i in range(overlap, num_samps, seq_len):
                x = quantize(batch[:, i-overlap : i+seq_len], q_type, q_levels)
                y = x[:, overlap : overlap+seq_len]
                yield (x, y)

    def get_steps_per_batch():
        (samples, _) = librosa.load(files[0], sr=None, mono=True)
        return int(np.floor(len(samples) / float(seq_len)))

    def get_steps_per_epoch():
        num_batches = dataset_size // args.batch_size
        steps_per_batch = get_steps_per_batch()
        return num_batches * steps_per_batch

    steps_per_epoch = get_steps_per_epoch()

    class TrainingStepCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.steps_per_epoch = get_steps_per_epoch()
            self.steps_per_batch = get_steps_per_batch()
            self.step_start_time = 0

        def on_train_begin(self, logs):
            latest_checkpoint = get_latest_checkpoint(logdir)
            if args.resume==True and latest_checkpoint:
                model.load_weights(latest_checkpoint)

        def on_epoch_begin(self, epoch, logs):
            self.epoch = epoch + 1

        def on_batch_begin(self, batch, logs):
            if (batch + 1) % self.steps_per_batch == 1 : model.reset_states()
            self.step_start_time = time.time()

        def on_batch_end(self, batch, logs):
            loss, acc = logs.get('loss'), logs.get('accuracy')
            step_duration = time.time() - self.step_start_time
            template = 'Epoch: {:d}/{:d}, Step: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f}, ({:.3f} sec/step)'
            end_char = '\r' if (args.verbose == False) and (batch + 1 != self.steps_per_epoch) else '\n'
            stats_string = template.format(self.epoch, args.num_epochs, batch + 1, self.steps_per_epoch, loss, acc * 100, step_duration)
            print(stats_string, end=end_char)


    checkpoint_path = '{0}/model.ckpt'.format(rundir) \
        if args.checkpoint_policy=='Best' \
        else '{0}/model.ckpt-{{epoch}}'.format(rundir)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=args.checkpoint_policy=='Best',
            save_freq=args.checkpoint_every * get_steps_per_epoch()),
        TrainingStepCallback(),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.TensorBoard(log_dir=rundir, update_freq=50)
    ]

    reduce_lr_after = args.reduce_learning_rate_after

    if reduce_lr_after is not None and reduce_lr_after > 0:
        def scheduler(epoch, leaning_rate):
            if epoch < reduce_lr_after:
                return leaning_rate
            else:
                return leaning_rate * tf.math.exp(-0.1)
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(scheduler)
        )

    # Train
    init_data = np.random.randint(0, model.q_levels, (model.batch_size, overlap + model.seq_len, 1))
    model(init_data)
    try:
        model.fit(
            train_iter(),
            epochs=args.num_epochs,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )
    except KeyboardInterrupt:
        print()
        print('Keyboard interrupt')
        print()


if __name__ == '__main__':
    main()
