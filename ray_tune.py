import tensorflow as tf
from hyperopt import hp
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
import numpy as np
import argparse
import librosa

from samplernn import SampleRNN
from dataset import (get_dataset, find_files, get_dataset_filenames_split)
from train import optimizer_factory


def get_arguments():

    def check_bool(value):
        val = str(value).upper()
        if 'TRUE'.startswith(val):
            return True
        elif 'FALSE'.startswith(val):
            return False
        else:
           raise ValueError('Argument is neither `True` nor `False`')

    parser = argparse.ArgumentParser(description='PRiSM SampleRNN Model Tuner')
    parser.add_argument('--data_dir',                   type=str,   required=True,
                                                        help='Path (absolute) to the directory containing the training data.')
    parser.add_argument('--id',                         type=str,   default='default', help='Id for the model tuning session')
    parser.add_argument('--logdir',                     type=str,   help='Path (absolute) to the directory to store results.')
    parser.add_argument('--verbose',                    type=check_bool,   default=True, help='Model training verbosity')
    parser.add_argument('--num_trials',                 type=int,   default=1, help='Number of trials (number of times to sample the search space)')
    parser.add_argument('--num_epochs',                 type=int,   default=30, help='Number of training epochs')
    parser.add_argument('--type',                       type=str,   default='bayesian', choices=['bayesian', 'random_search'],
                                                        help='Type of tuning algorithm to use, either Bayesian Optimization or Random Search.')
    #parser.add_argument('--metric',                     type=str,   default='val_loss', choices=['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                                                        #help='Metric to set as the objective for tuning')
    parser.add_argument('--num_cpus',                   type=int,   default=1, help='Number of CPUs to use')
    parser.add_argument('--num_gpus',                   type=int,   default=0, help='Number of GPUs to use')
    parser.add_argument('--early_stopping_patience',    type=int,   default=3,
                                                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--val_frac',                   type=float, default=0.1,
                                                        help='Fraction of the dataset to reserve for validaton. Will be rounded to the closest multiple of the batch size.')
    parser.add_argument('--frame_sizes',                type=int,   required=True, nargs='+', action='append',
                                                        help='Frame sizes (in samples) of the two upper tiers in the model, in ascending order. Note that the frame size of the upper tier must be an even multiple of that of the lower tier')
    parser.add_argument('--batch_size',                 type=int,   required=True, nargs='+', help='Size of the mini-batch')
    parser.add_argument('--seq_len',                    type=int,   required=True, nargs='+', help='RNN sequence length')
    parser.add_argument('--dim',                        type=int,   default=[1024, 2048], nargs='+', help='RNN output space dimensionality')
    parser.add_argument('--rnn_type',                   type=str,   default=['gru', 'lstm'], nargs='+', help='RNN type (GRU or LSTM)')
    parser.add_argument('--num_rnn_layers',             type=int,   default=[1, 2, 4, 8], nargs='+', help='Number of RNN layers')
    parser.add_argument('--q_type',                     type=str,   default=['mu-law', 'linear'], nargs='+', help='Quantization type')
    parser.add_argument('--rnn_dropout',                type=float, default=[0.2, 0.4, 0.6], nargs='+', help='Size of the RNN dropout')
    parser.add_argument('--optimizer',                  type=str,   default='adam', choices=optimizer_factory.keys(), help='Type of training optimizer to use')
    parser.add_argument('--learning_rate',              type=float, default=[1e-2, 1e-3, 1e-4], nargs='+', help='Learning rate of training')
    parser.add_argument('--momentum',                   type=float, default=[0.1, 0.5, 0.9], nargs='+', help='Optimizer momentum')
    return parser.parse_args()

args = get_arguments()

search_space = {
    'batch_size' : hp.choice('batch_size', args.batch_size),
    'frame_sizes' : hp.choice('frame_sizes', args.frame_sizes),
    'seq_len' : hp.choice('seq_len', args.seq_len),
    'q_type' : hp.choice('q_type', args.q_type),
    'dim' : hp.choice('dim', args.dim),
    'rnn_type' : hp.choice('rnn_type', args.rnn_type),
    'num_rnn_layers' : hp.choice('num_rnn_layers', args.num_rnn_layers),
    'rnn_dropout' : hp.choice('rnn_dropout', args.rnn_dropout),
    'skip_conn' : hp.choice('skip_conn', [True, False]),
    'learning_rate' : hp.choice('learning_rate', args.learning_rate),
    'momentum' : hp.choice('momentum', args.momentum)
}

def train(config):
    # Create and compile the model
    model = SampleRNN(
        batch_size=config['batch_size'],
        frame_sizes=config['frame_sizes'],
        seq_len=config['seq_len'],
        q_type=config['q_type'],
        q_levels=256,
        dim=config['dim'],
        rnn_type=config['rnn_type'],
        num_rnn_layers=config['num_rnn_layers'],
        emb_size=256,
        skip_conn=config['skip_conn'],
        #skip_conn=False,
        rnn_dropout=config['rnn_dropout']
    )
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=config['learning_rate'],
        momentum=config['momentum']
    )
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=optimizer, loss=compute_loss, metrics=[train_accuracy])
    
    num_epochs = args.num_epochs
    batch_size = model.batch_size
    seq_len = model.seq_len
    overlap = model.big_frame_size
    q_type = model.q_type
    q_levels = 256

    (train_split, val_split) = get_dataset_filenames_split(
        args.data_dir, args.val_frac, model.batch_size)

    # Train and validation datasets
    train_dataset = get_dataset(train_split, num_epochs, batch_size, seq_len, overlap,
                                drop_remainder=True, q_type=q_type, q_levels=q_levels)
    val_dataset = get_dataset(val_split, 1, batch_size, seq_len, overlap, shuffle=False,
                                drop_remainder=True, q_type=q_type, q_levels=q_levels)

    # Get subseqs per batch...
    samples0, _ = librosa.load(train_split[0], sr=None, mono=True)
    steps_per_batch = int(np.floor(len(samples0) / float(seq_len)))

    # Get subseqs per epoch...
    steps_per_epoch = len(train_split) // batch_size * steps_per_batch

    verbose = 1 if args.verbose==True else 2

    # Train...
    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        shuffle=False,
        validation_data=val_dataset,
        callbacks=[TuneReporter()],
        verbose=verbose
    )

class TuneReporter(tf.keras.callbacks.Callback):
    """Tune Reporter Callback."""

    def __init__(self, reporter=None, freq="epoch", logs=None):
        """Initializer.
        Args:
            freq (str): Sets the frequency of reporting intermediate results.
        """
        self.iteration = 0
        logs = logs or {}
        self.freq = freq
        super(TuneReporter, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if not self.freq == "epoch":
            return
        self.iteration += 1
        if "acc" in logs:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs.get("accuracy"))

if __name__ == "__main__":

    ray.init(num_gpus=args.num_gpus)

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="val_loss",
        max_t=400,
        grace_period=20)
        
    search = HyperOptSearch(
        space=search_space,
        metric="val_loss",
        mode="min")

    analysis = tune.run(
        train,
        name=args.id,
        local_dir=args.logdir,
        scheduler=sched,
        search_alg=search,
        mode="min",
        num_samples=args.num_trials,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1
        })
    print('\n')
    print("Best Hparams:", analysis.get_best_config(
        metric="val_loss"
    ))
    print('\n')
    print("Results Summary:", analysis.get_best_trial(
        metric="val_loss"
    ))

# Run it like:
'''
# Have to specify data_dir as an absolute path because
# of https://github.com/ray-project/ray/issues/9571.
nohup python ray_tune.py \
  --data_dir /full/path/to/dataset \
  --num_epochs 2 \
  --frame_sizes 16 64 \
  --frame_sizes 32 128 \
  --batch_size 16 32 64 \
  --seq_len 512 1024 2048 \
  --num_rnn_layers 2 4 \
  > tuner.log 2>&1 </dev/null &
'''
