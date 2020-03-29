import tensorflow as tf
import numpy as np
import os
import sys
import time
import samplernn as srnn
import dataset as ds

from importlib import reload

# To run the original training code:
# python train.py --data_dir=./test_chunks --silence_threshold=0.1 --sample_size=102408 --big_frame_size=8 --frame_size=2 --q_levels=256 --rnn_type=GRU --dim=1024 --n_rnn=1 --seq_len=520 --emb_size=256 --batch_size=1 --optimizer=adam

# To run train.py:
# python train.py --data_dir ./22k_full --id test --num_epochs 2 --batch_size 1 --max_checkpoints 2 --output_file_dur 1 --sample_rate 22050


NUM_EPOCHS = 1
Q_LEVELS = 256
BIG_FRAME_SIZE = 8
FRAME_SIZE = 2

def get_test_input(data_dir, model):
    audio = srnn.load_audio(data_dir)
    batches = np.array([next(audio)])
    input_seq_len = model.seq_len + model.big_frame_size
    return srnn.mu_law_encode(batches[:, :input_seq_len, :], Q_LEVELS)

def test_model(data_dir='./kalamata_you_chunks', frame_sizes=[FRAME_SIZE, BIG_FRAME_SIZE], seq_len=1024):
    model = srnn.SampleRNN(frame_sizes=frame_sizes, batch_size=1, q_levels=256, dim=1024, num_rnn_layers=1, seq_len=seq_len, emb_size=256)
    encoded_inputs_rnn = get_test_input(data_dir, model)
    return model(encoded_inputs_rnn)

def test_conv_transpose(input_size=1024, dim=1024, big_frame_size=64, frame_size=16):
    batch_size = 1
    num_lower_tier_frames = big_frame_size // frame_size
    num_steps = input_size // big_frame_size
    kernel_size = num_lower_tier_frames
    print(num_lower_tier_frames, num_steps, kernel_size)
    data = np.random.rand(batch_size * num_steps * dim)
    frame_outputs = tf.constant(data, shape=[batch_size, num_steps, dim], dtype=tf.float32)
    kernel = tf.constant(
        tf.initializers.GlorotNormal()(shape=[kernel_size, dim, dim]),
        dtype=tf.float32)
    return tf.nn.conv1d_transpose(
        input=frame_outputs,
        filters=kernel,
        output_shape=[batch_size, num_steps * kernel_size, dim],
        strides=kernel_size)


def test_training(data_dir='./test_chunks', num_steps=20, batch_size=1, seq_len=1024):
    logdir = './logdir/train/test'
    model = srnn.SampleRNN(frame_sizes=[2,8], batch_size=batch_size, q_levels=Q_LEVELS, dim=1024, num_rnn_layers=1, seq_len=seq_len, emb_size=256)
    overlap = model.big_frame_size
    dataset = ds.get_dataset(data_dir, batch_size, seq_len, overlap)
    opt = tf.optimizers.Adam(learning_rate=1e-3, epsilon=1e-4)
    train_iter = iter(dataset)

    checkpoint_prefix = os.path.join(logdir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
    writer = tf.summary.create_file_writer(logdir)

    @tf.function
    def train_step(inputs):
        print('Tracing!')
        tf.print('Executing')
        loss_sum = 0
        final_big_frame_state = tf.zeros([batch_size, model.dim], tf.float32)
        final_frame_state = tf.zeros([batch_size, model.dim], tf.float32)
        for i in range(0, 10):
            #tf.print(loss_sum)
            with tf.GradientTape() as tape:
                seq = inputs[:, i : i+seq_len, :]
                encoded_inputs_rnn = srnn.mu_law_encode(seq, Q_LEVELS)
                encoded_rnn = srnn.one_hot_encode(encoded_inputs_rnn, batch_size, Q_LEVELS)
                (raw_output, final_big_frame_state, final_frame_state) = model(
                    encoded_inputs_rnn,
                    final_big_frame_state,
                    final_frame_state,
                    training=True,
                )
                target = tf.reshape(encoded_rnn[:, BIG_FRAME_SIZE:, :], [-1, Q_LEVELS])
                prediction = tf.reshape(raw_output, [-1, Q_LEVELS])
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=tf.argmax(target, axis=-1)))
                loss_sum += loss #/ rnn_len
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(list(zip(grads, model.trainable_variables)))
        return loss_sum

    total_loss = 0
    for step in range(num_steps):
        inputs = next(train_iter)
        tf.summary.trace_on(graph=True, profiler=True)
        total_loss += train_step(inputs)
        with writer.as_default():
            tf.summary.trace_export(
                name="samplernn_model_trace",
                step=step,
                profiler_outdir=logdir
            )
        print()
        print(total_loss)

        if step % 5 == 0:
            checkpoint.save(checkpoint_prefix)
            print('Storing checkpoint to {} ...'.format(logdir), end="")
            sys.stdout.flush()

# It seems calling tf.summary inside a function decorated by tf.function does not work,
# contrary to the documentation here: https://www.tensorflow.org/api_docs/python/tf/summary
def tffunc_summary_test():
    writer = tf.summary.create_file_writer("./logdir/train/test2")

    @tf.function
    def test_func(step):
        with writer.as_default():
            tf.summary.scalar("test_metric", 0.5, step=step)

    for step in range(100):
        test_func(step)
        writer.flush()
    

def test_subbatcher(dataset, seq_len=1024):
    for batch in dataset:
        num_samps = len(batch[0])
        for i in range(0, num_samps, seq_len):
            time.sleep(1)
            print()
            print(i/seq_len)
            sequences = batch[:, i : i+seq_len]
            yield sequences


def test_training2(data_dir='./test_chunks', batch_size=1, seq_len=1024, logdir='./logdir/train/test2'):
    model = srnn.SampleRNN(frame_sizes=[2,8], batch_size=batch_size, q_levels=Q_LEVELS, dim=1024, num_rnn_layers=1, seq_len=seq_len, emb_size=256)
    overlap = model.big_frame_size
    dataset = ds.get_dataset(data_dir, batch_size, seq_len, overlap)
    opt = tf.optimizers.Adam(learning_rate=1e-3, epsilon=1e-4)
    
    def train_iter():
        for batch in dataset:
            reset = True
            num_samps = len(batch[0])
            for i in range(0, num_samps, seq_len):
                sequences = batch[:, i : i+seq_len+overlap]
                yield (sequences, reset)
                reset = False

    checkpoint_prefix = os.path.join(logdir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)

    @tf.function
    def train_step(inputs):
        print('Tracing!')
        tf.print('Executing')
        (inputs, reset) = inputs
        final_big_frame_state = tf.zeros([batch_size, model.dim], tf.float32) if reset==True else None
        final_frame_state = tf.zeros([batch_size, model.dim], tf.float32) if reset==True else None
        with tf.GradientTape() as tape:
            encoded_inputs_rnn = srnn.mu_law_encode(inputs, Q_LEVELS)
            encoded_rnn = srnn.one_hot_encode(encoded_inputs_rnn, batch_size, Q_LEVELS)
            (raw_output, final_big_frame_state, final_frame_state) = model(
                encoded_inputs_rnn,
                final_big_frame_state,
                final_frame_state,
                training=True,
            )
            target = tf.reshape(encoded_rnn[:, BIG_FRAME_SIZE:, :], [-1, Q_LEVELS])
            prediction = tf.reshape(raw_output, [-1, Q_LEVELS])
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=prediction,
                    labels=tf.argmax(target, axis=-1)))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(list(zip(grads, model.trainable_variables)))
        return loss

    for (step, inputs) in enumerate(train_iter()):
        loss = train_step(inputs)
        print()
        print(loss)

        with writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            #writer.flush() # But see https://stackoverflow.com/a/52502679

        if step % 20 == 0:
            checkpoint.save(checkpoint_prefix)
            print('Storing checkpoint to {} ...'.format(logdir), end="")
            sys.stdout.flush()
        
        if step == 0:
            with writer.as_default():
                tf.summary.trace_export(
                    name="samplernn_model_trace_2.1",
                    step=0,
                    profiler_outdir=logdir)



def get_arguments_test():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--frame_sizes', type=int, nargs='*', default=[16, 64])
    parsed_args = parser.parse_args()
    assert parsed_args.seq_len % parsed_args.frame_sizes[1] == 0,\
        'seq_len should be evenly divisible by tier 2 frame size'
    assert parsed_args.frame_sizes[1] % parsed_args.frame_sizes[0] == 0,\
        'Tier 2 frame size should be evenly divisible by tier 1 frame size'
    return parsed_args

###############################################################################

def pad_batch(batch, batch_size, seq_len, overlap):
    num_samps = len(batch[0])
    padding = ( seq_len - 1 - (num_samps + seq_len - 1) % seq_len ) + overlap
    padded_batch = np.zeros([batch_size, num_samps + padding, 1], dtype='float32')
    for (i, samples) in enumerate(batch):
        padded_batch[i, :len(samples), :] = samples
    return padded_batch

def get_epoch_dataset(data_dir, batch_size, seq_len, overlap):
    dataset = tf.data.Dataset.from_generator(
        # See here for why lambda is required: https://stackoverflow.com/a/56602867
        lambda: srnn.load_audio(data_dir),
        output_types=tf.float32,
        output_shapes=((None, 1)), # Not sure about the precise value of this...
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda batch: tf.py_function(
        func=pad_batch, inp=[batch, batch_size, seq_len, overlap], Tout=tf.float32
    ))
    return dataset

def epoch_test(data_dir='./test_chunks', num_epochs=2, batch_size=1, seq_len=65536, overlap=256):
    dataset = get_epoch_dataset(data_dir, batch_size, seq_len, overlap)

    def test_iter():
        for batch in dataset:
            reset = True
            num_samps = len(batch[0])
            for i in range(overlap, num_samps, seq_len):
                seqs = batch[:, i-overlap : i+seq_len]
                yield (seqs, reset)
                reset = False

    template = 'epoch: {}, step: {}, batch: {}, reset: {}'
    for epoch in range(num_epochs):
        batch = -1
        for (step, (inputs, reset)) in enumerate(test_iter()):
            if reset==True:
                batch += 1
            print(template.format(epoch, step, batch, reset))

