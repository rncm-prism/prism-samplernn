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
CONFIG_FILE = './default.json'
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
    parser.add_argument('--id',                         type=str,            required=True,
                                                        help='Id for the current training session')
    parser.add_argument('--verbose',                    type=check_bool,
                                                        help='Whether to print training step output to a new line each time (the default), or overwrite the last output')
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
    return tf.optimizers.Adam(learning_rate=learning_rate)


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
        num_rnn_layers=config.get('num_rnn_layers'),
        emb_size=config.get('emb_size'),
        skip_conn=config.get('skip_conn')
    )


def main():
    args = get_arguments()

    # Create training session directories
    if not find_files(args.data_dir):
        raise ValueError("No audio files found in '{}'.".format(args.data_dir))
    logdir_train = os.path.join(args.logdir_root, args.id, 'train')
    if not os.path.exists(logdir_train):
        os.makedirs(logdir_train)
    logdir_predict = os.path.join(args.logdir_root, args.id, 'predict')
    if not os.path.exists(logdir_predict):
        os.makedirs(logdir_predict)
    generate_dir = os.path.join(args.output_dir, args.id)
    if not os.path.exists(generate_dir):
        os.makedirs(generate_dir)

    # Load model configuration
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)
    model = create_model(args.batch_size, config)
    q_type = model.q_type
    q_levels = model.q_levels

    # Optimizer
    opt = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    overlap = model.big_frame_size
    dataset = get_dataset(args.data_dir, args.batch_size, model.seq_len, overlap)

    # Dataset iterator
    def train_iter():
        seq_len = model.seq_len
        for batch in dataset:
            reset = True
            num_samps = len(batch[0])
            for i in range(overlap, num_samps, seq_len):
                seqs = batch[:, i-overlap : i+seq_len]
                yield (seqs, reset)
                reset = False

    def get_steps_per_epoch():
        files = find_files(args.data_dir)
        num_batches = len(files) // args.batch_size
        (samples, _) = librosa.load(files[0], sr=None, mono=False)
        steps_per_batch = int(np.floor(len(samples) / float(model.seq_len)))
        return num_batches * steps_per_batch

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1), optimizer=opt, model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=logdir_train, max_to_keep=args.max_checkpoints)
    writer = tf.summary.create_file_writer(logdir_train)
    tf.summary.trace_on(graph=True, profiler=True)

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    # Training step function
    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            inputs = quantize(inputs, q_type, q_levels)
            raw_output = model(
                inputs,
                training=True)
            prediction = tf.reshape(raw_output, [-1, q_levels])
            target = tf.reshape(inputs[:, model.big_frame_size:, :], [-1])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction,
                    labels=target))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(list(zip(grads, model.trainable_variables)))
        train_accuracy.update_state(target, prediction)
        return loss

    # Maybe resume
    if args.resume==True and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    training_start_time = time.time()
    epoch_start = ckpt.epoch.numpy()
    # Track previous epoch so non-verbose output can
    # be written to a new line when changing epoch
    prev_epoch = 0

    steps_per_epoch = get_steps_per_epoch()

    # Training loop
    try:
        for epoch in range(epoch_start, args.num_epochs + 1):
            ckpt.epoch.assign(epoch)
            batch = -1
            for (step, (inputs, reset)) in enumerate(train_iter()):
                # Reset RNN states at the end of each batch
                if reset==True:
                    batch += 1
                    if batch > 0: model.reset_states()

                step_start_time = time.time()

                # Compute training loss and accuracy
                loss = train_step(inputs)
                train_acc = train_accuracy.result() * 100

                # Print step stats
                step_duration = time.time() - step_start_time
                template = 'Epoch: {:d}/{:d}, Step: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f}, ({:.3f} sec/step)'
                end_char = '\r' if (args.verbose == False) and (epoch != prev_epoch) else '\n'
                print(template.format(epoch, args.num_epochs, step, steps_per_epoch, loss, train_acc, step_duration), end=end_char)

                # Write summaries
                with writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)
                    tf.summary.scalar('accuracy', train_acc, step=step)

                # Checkpoint
                if step % args.checkpoint_every == 0:
                    ckpt_manager.save()

                if epoch == 1 and step == 0:
                    with writer.as_default():
                        tf.summary.trace_export(
                            name="samplernn_model_trace",
                            step=0,
                            profiler_outdir=logdir_train)

            train_accuracy.reset_states()

            prev_epoch = epoch

            time_elapsed = time.time() - training_start_time
            print('Time elapsed since start of training: {:.3f} seconds'.format(time_elapsed))

            # Save model weights for inference model
            print('Saving checkpoint for epoch {}...'.format(epoch))
            epoch_ckpt_path = '{}/ckpt-{}'.format(logdir_predict, epoch)
            model.save_weights(epoch_ckpt_path)

            # Generate samples
            print('Generating samples for epoch {}...'.format(epoch))
            output_file_path = '{}/{}_epoch_{}.wav'.format(generate_dir, args.id, epoch)
            generate(output_file_path, epoch_ckpt_path, config, args.max_generate_per_epoch, args.output_file_dur,
                    args.sample_rate, args.temperature, args.seed, args.seed_offset)

    except KeyboardInterrupt:
        print()
        print('Keyboard interrupt')
        print()


if __name__ == '__main__':
    main()
