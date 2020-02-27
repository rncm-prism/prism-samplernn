from __future__ import print_function
import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from samplernn import SampleRNN
from samplernn import (load_audio, write_wav, find_files)
from samplernn import (mu_law_encode, mu_law_decode)
from samplernn import (optimizer_factory, one_hot_encode, unsqueeze)

from dataset import get_dataset


LOGDIR_ROOT = './logdir'
OUTDIR = './generated'
NUM_EPOCHS = 100
BATCH_SIZE = 1
BIG_FRAME_SIZE = 64
FRAME_SIZE = 16
SEQ_LEN = 1024
Q_LEVELS = 256
Q_ZERO = Q_LEVELS // 2
DIM = 1024
N_RNN = 1
EMB_SIZE = 256
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = None
OUTPUT_DUR = 3 # Duration of generated audio in seconds
CHECKPOINT_EVERY = 5
MAX_CHECKPOINTS = 5
GENERATE_EVERY = 10
SAMPLE_RATE = 44100 # Sample rate of generated audio
MAX_GENERATE_PER_BATCH = 10
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
    parser.add_argument('--batch_size',                 type=check_positive, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--logdir_root',                type=str,            default=LOGDIR_ROOT,
                                                        help='Root directory for training log files')
    parser.add_argument('--output_dir',                 type=str,            default=OUTDIR,
                                                        help='Path to the directory for generated audio')
    parser.add_argument('--output_file_dur',            type=str,            default=OUTPUT_DUR,
                                                        help='Duration of generated audio files')
    parser.add_argument('--sample_rate',                type=check_positive, default=SAMPLE_RATE,
                                                        help='Sample rate of the generated audio')
    parser.add_argument('--num_epochs',                 type=check_positive, default=NUM_EPOCHS,
                                                        help='Number of training epochs')
    parser.add_argument('--checkpoint_every',           type=check_positive, default=CHECKPOINT_EVERY)
    parser.add_argument('--learning_rate',              type=float,          default=LEARNING_RATE)
    parser.add_argument('--l2_regularization_strength', type=float,          default=L2_REGULARIZATION_STRENGTH)
    parser.add_argument('--silence_threshold',          type=float,          default=SILENCE_THRESHOLD)
    parser.add_argument('--optimizer',                  type=str,            default='adam', choices=optimizer_factory.keys(),
                                                        help='Type of training optimizer to use')
    parser.add_argument('--momentum',                   type=float,          default=MOMENTUM)
    parser.add_argument('--seq_len',                    type=check_positive, default=SEQ_LEN,
                                                        help='Number of samples in each truncated BPTT pass')
    parser.add_argument('--frame_sizes',                type=int,            default=[FRAME_SIZE, BIG_FRAME_SIZE], nargs='*',
                                                        help='Number of samples per frame in each tier')
    #parser.add_argument('--q_levels',                   type=int,            default=Q_LEVELS, help='Number of audio quantization bins')
    parser.add_argument('--dim',                        type=check_positive, default=DIM,
                                                        help='Number of cells in every RNN and MLP layer')
    parser.add_argument('--n_rnn',                      type=check_positive, default=N_RNN, choices=list(range(1, 6)),
                                                        help='Number of RNN layers in each tier')
    parser.add_argument('--emb_size',                   type=check_positive, default=EMB_SIZE,
                                                        help='Size of the embedding layer')
    parser.add_argument('--max_checkpoints',            type=check_positive, default=MAX_CHECKPOINTS,
                                                        help='Maximum number of training checkpoints to keep')
    parser.add_argument('--resume',                     type=check_bool,     default=RESUME,
                                                        help='Whether to resume training from the last available checkpoint')
    parser.add_argument('--val_pcnt',                   type=float,          default=VAL_PCNT,
                                                        help='Percentage of data to reserve for validation')
    parser.add_argument('--test_pcnt',                  type=float,          default=TEST_PCNT,
                                                        help='Percentage of data to reserve for testing')
    parsed_args = parser.parse_args()
    assert parsed_args.frame_sizes[0] < parsed_args.frame_sizes[1], 'Frame sizes should be specified in ascending order'
    # The following parameter interdependencies are sourced from the original implementation:
    # https://github.com/soroushmehr/sampleRNN_ICLR2017/blob/master/models/three_tier/three_tier.py
    assert parsed_args.seq_len % parsed_args.frame_sizes[1] == 0,\
        'seq_len should be evenly divisible by tier 2 frame size'
    assert parsed_args.frame_sizes[1] % parsed_args.frame_sizes[0] == 0,\
        'Tier 2 frame size should be evenly divisible by tier 1 frame size'
    return parsed_args


def generate(model, step, dur, sample_rate, outdir):
    num_samps = dur * sample_rate
    samples = np.zeros((model.batch_size, num_samps, 1), dtype='int32')
    samples[:, :model.big_frame_size, :] = Q_ZERO
    q_vals = np.arange(Q_LEVELS)
    progress_every = 250
    start_time = time.time()
    for t in range(model.big_frame_size, num_samps):
        if t % model.big_frame_size == 0:
            inputs = samples[:, t - model.big_frame_size : t, :].astype('float32')
            big_frame_outputs = model.big_frame_rnn(
                inputs,
                num_steps=1)
        if t % model.frame_size == 0:
            inputs = samples[:, t - model.frame_size : t, :].astype('float32')
            big_frame_output_idx = (t // model.frame_size) % (
                model.big_frame_size // model.frame_size
            )
            frame_outputs = model.frame_rnn(
                inputs,
                num_steps=1,
                conditioning_frames=unsqueeze(big_frame_outputs[:, big_frame_output_idx, :], 1))
        inputs = samples[:, t - model.frame_size : t, :]
        frame_output_idx = t % model.frame_size
        sample_outputs = model.sample_mlp(
            inputs,
            conditioning_frames=unsqueeze(frame_outputs[:, frame_output_idx, :], 1))
        sample_outputs = tf.cast(
            tf.reshape(sample_outputs, [-1, Q_LEVELS]),
            tf.float64
        )
        generated = []
        for row in tf.cast(tf.nn.softmax(sample_outputs), tf.float32):
            samp = np.random.choice(q_vals, p=row)
            generated.append(samp)
        start = t - model.big_frame_size
        if start % progress_every == 0:
            end = min(start + progress_every, num_samps)
            duration = time.time() - start_time
            print('Generating samples {} - {} of {} (time elapsed: {:.3f})'.format(start+1, end, num_samps, duration))
        samples[:, t] = np.array(generated).reshape([-1, 1])
    template = '{}/step_{}.{}.wav'
    for i in range(model.batch_size):
        samples = samples[i].reshape([-1, 1]).tolist()
        audio = mu_law_decode(samples, Q_LEVELS)
        path = template.format(outdir, str(step), str(i))
        write_wav(path, audio, sample_rate)
        print('Generated sample output to {}'.format(path))
        if i >= MAX_GENERATE_PER_BATCH: break
    print('Done')


def maybe_resume(ckpt_manager, ckpt, logdir):
    print("Attempting to restore saved checkpoints ...", end="")
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt:
        #print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        print("  Restoring...", end="")
        print(" Done.")
        return ckpt
    else:
        print(" No saved checkpoints found, starting new training session")
        return None


def main():
    args = get_arguments()
    if not find_files(args.data_dir):
        raise ValueError("No audio files found in '{}'.".format(args.data_dir))
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    logdir = os.path.join(args.logdir_root, 'train')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    model = SampleRNN(
        batch_size=args.batch_size,
        frame_sizes=args.frame_sizes,
        q_levels=Q_LEVELS, #args.q_levels,
        dim=args.dim,
        n_rnn=args.n_rnn,
        seq_len=args.seq_len,
        emb_size=args.emb_size,
    )
    opt = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum,
    )

    overlap = model.big_frame_size
    dataset = get_dataset(args.data_dir, args.batch_size, args.seq_len, overlap)

    def train_iter():
        seq_len = args.seq_len
        for batch in dataset:
            reset = True
            num_samps = len(batch[0])
            for i in range(overlap, num_samps, seq_len):
                seqs = batch[:, i-overlap : i+seq_len]
                yield (seqs, reset)
                reset = False

    ckpt = tf.train.Checkpoint(optimizer=opt, model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=logdir, max_to_keep=args.max_checkpoints)
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            inputs = mu_law_encode(inputs, Q_LEVELS)
            encoded_rnn = one_hot_encode(inputs, args.batch_size, Q_LEVELS)
            raw_output = model(
                inputs,
                training=True,
            )
            target = tf.reshape(encoded_rnn[:, model.big_frame_size:, :], [-1, Q_LEVELS])
            prediction = tf.reshape(raw_output, [-1, Q_LEVELS])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction,
                    labels=tf.argmax(target, axis=-1)))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt.apply_gradients(list(zip(grads, model.trainable_variables)))
        return loss

    if args.resume==True:
        ckpt.restore(ckpt_manager.latest_checkpoint)

    for epoch in range(args.num_epochs):
        batch = -1
        for (step, (inputs, reset)) in enumerate(train_iter()):
            if reset==True:
                batch += 1
                if batch > 0: model.reset_hidden_states()
                if (batch-1) % GENERATE_EVERY == 0 and batch > GENERATE_EVERY:
                    print('Generating samples...')
                    #generate_and_save_samples(model, step, args.output_file_dur, args.sample_rate, args.output_dir)

            start_time = time.time()
            loss = train_step(inputs)

            duration = time.time() - start_time
            template = 'Epoch: {:d}, Step: {:d}, Loss: {:.3f}, ({:.3f} sec/step)'
            print(template.format(epoch, step, loss, duration))

            with writer.as_default():
                tf.summary.scalar('loss', loss, step=step)

            if step % args.checkpoint_every == 0:
                ckpt_manager.save()
                sys.stdout.flush()
            
            if epoch == 0 and step == 0:
                with writer.as_default():
                    tf.summary.trace_export(
                        name="samplernn_model_trace",
                        step=0,
                        profiler_outdir=logdir)


if __name__ == '__main__':
    main()
