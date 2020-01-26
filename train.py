from __future__ import print_function
import argparse
import os
import sys
import time

import librosa
import numpy as np
import tensorflow as tf

from samplernn import SampleRNN
from samplernn import (load_audio, find_files)
from samplernn import mu_law_decode
from samplernn import optimizer_factory

LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 5
GENERATE_EVERY = 10
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = None
MOMENTUM = 0.9
MAX_TO_KEEP = 5
N_SECS = 3
SAMPLE_RATE = 44100 #22050
LENGTH = N_SECS * SAMPLE_RATE
SEQ_LEN = 1024
Q_LEVELS = 256
DIM = 1024
N_RNN = 1
BATCH_SIZE = 64 #1
NUM_GPUS = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='SampleRnn')
    parser.add_argument('--data_dir',                   type=str,   required=True,
                                                        help='Path to the directory containing the training data')
    parser.add_argument('--num_gpus',                   type=int,   default=NUM_GPUS, help='Number of GPUs')
    parser.add_argument('--batch_size',                 type=int,   default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--logdir_root',                type=str,   default=LOGDIR_ROOT,
                                                        help='Root directory for training log files')
    parser.add_argument('--checkpoint_every',           type=int,   default=CHECKPOINT_EVERY)
    parser.add_argument('--num_steps',                  type=int,   default=NUM_STEPS)
    parser.add_argument('--learning_rate',              type=float, default=LEARNING_RATE)
    parser.add_argument('--sample_size',                type=int,   default=SAMPLE_SIZE)
    parser.add_argument('--sample_rate',                type=int,   default=SAMPLE_RATE,
                                                        help='Sample rate of the training data and generated audio')
    parser.add_argument('--l2_regularization_strength', type=float, default=L2_REGULARIZATION_STRENGTH)
    parser.add_argument('--silence_threshold',          type=float, default=SILENCE_THRESHOLD)
    parser.add_argument('--optimizer',                  type=str,   default='adam', choices=optimizer_factory.keys(),
                                                        help='Type of training optimizer to use')
    parser.add_argument('--momentum',                   type=float, default=MOMENTUM)
    parser.add_argument('--seq_len',                    type=int,   default=SEQ_LEN,
                                                        help='Number of samples in each truncated BPTT pass')
    parser.add_argument('--frame_sizes',                type=int,   required=True, nargs='*',
                                                        help='Frame sizes in terms of the number of lower tier frames')
    parser.add_argument('--q_levels',                   type=int,   default=Q_LEVELS,
                                                        help='Number of audio quantization bins')
    parser.add_argument('--dim',                        type=int,   default=DIM,
                                                        help='Number of cells in every RNN and MLP layer')
    parser.add_argument('--n_rnn',                      type=int,   default=N_RNN, choices=list(range(1, 6)),
                                                        help='Number of RNN layers in each tier')
    parser.add_argument('--emb_size',                   type=int,   required=True)
    parser.add_argument('--max_checkpoints',            type=int,   default=MAX_TO_KEEP)
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def main():
    args = get_arguments()
    if not find_files(args.data_dir):
        raise ValueError("No audio files found in '{}'.".format(args.data_dir))
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    logdir = os.path.join(args.logdir_root, 'train')
    
    dist_strategy = tf.distribute.MirroredStrategy()
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: load_audio(args.data_dir, args.sample_rate, args.sample_size, args.silence_threshold),
        output_types=tf.float32,
        output_shapes=((None, 1)), # Not sure about the precise value of this...
    )
    dataset = dataset.batch(args.batch_size)
    #[tf.print(batch) for batch in dataset]
    dist_dataset = dist_strategy.experimental_distribute_dataset(dataset)
    with dist_strategy.scope()
        model = SampleRNN(
            batch_size=args.batch_size,
            frame_sizes=args.frame_sizes,
            q_levels=args.q_levels,
            dim=args.dim,
            n_rnn=args.n_rnn,
            seq_len=args.seq_len,
            emb_size=args.emb_size,
        )
        optim = optimizer_factory[args.optimizer](
            learning_rate=args.learning_rate,
            momentum=args.momentum,
        )
        checkpoint = tf.train.Checkpoint(optimizer=optim, model=model)
        writer = tf.summary.create_file_writer(logdir)

    def step_fn(inputs):
        inputs = inputs[args.batch_size, args.seq_len, 1]
        encoded_inputs_rnn = mu_law_encode(inputs, args.q_levels)
        encoded_rnn = model._one_hot(encoded_input_rnn)
        with tf.GradientTape() as tape:
            raw_output, final_big_frame_state, final_frame_state = model(
                encoded_inputs_rnn,
                training=True,
            )
            target_output_rnn = encoded_rnn[:, BIG_FRAME_SIZE:, :]
            target_output_rnn = tf.reshape(
                target_output_rnn, [-1, args.q_levels])
            prediction = tf.reshape(raw_output, [-1, args.q_levels])
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction,
                labels=labels,
            )
            loss = tf.reduce_sum(cross_entropy) * (1.0 / args.batch_size)
            tf.summary.scalar('loss', loss)
            writer.flush() # But see https://stackoverflow.com/a/52502679
        grads = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(list(zip(grads, model.trainable_variables)))
        return cross_entropy

    with dist_strategy.scope()
        for inputs in dist_dataset:
            losses = dist_strategy.experimental_run_v2(
                step_fn,
                args=(inputs),
            )
            mean_loss = dist_strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                losses,
                axis=0,
            )


if __name__ == '__main__':
    main()
