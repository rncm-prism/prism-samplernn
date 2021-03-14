import tensorflow as tf
import kerastuner as kt
import collections
import numpy as np
import argparse
import librosa

from samplernn import SampleRNN
from dataset import (get_dataset, find_files, get_dataset_filenames_split)
from train import optimizer_factory


def get_arguments():
    parser = argparse.ArgumentParser(description='PRiSM SampleRNN Model Tuner')
    parser.add_argument('--data_dir',                   type=str,   required=True,
                                                        help='Path to the directory containing the training data.')
    parser.add_argument('--id',                         type=str,   default='default', help='Id for the model tuning session')
    parser.add_argument('--num_epochs',                 type=int,   default=30,
                                                        help='Number of training epochs')
    parser.add_argument('--type',                       type=str,   default='bayesian', choices=['bayesian', 'random_search'],
                                                        help='Type of tuning algorithm to use, either Bayesian Optimization or Random Search.')
    parser.add_argument('--objective',                  type=str,   default='val_loss', choices=['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                                                        help='Metric to set as the objective for tuning')
    parser.add_argument('--max_trials',                 type=int,   default=10, help='Maximum nuber of trials to run')
    parser.add_argument('--early_stopping_patience',    type=int,   default=3,
                                                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--val_frac',                   type=float, default=0.1,
                                                        help='Fraction of the dataset to be set aside for validation, rounded to the nearest multiple of the batch size. Defaults to 0.1, or 10%%.')
    parser.add_argument('--frame_sizes',                type=int,   required=True, nargs='+', action='append',
                                                        help='Frame sizes (in samples) of the two upper tiers in the model, in ascending order. Note that the frame size of the upper tier must be an even multiple of that of the lower tier')
    parser.add_argument('--batch_size',                 type=int,   required=True, nargs='+', help='Size of the mini-batch')
    parser.add_argument('--seq_len',                    type=int,   required=True, nargs='+', help='RNN sequence length')
    parser.add_argument('--dim',                        type=int,   default=[1024, 2048], nargs='+', help='RNN output space dimensionality')
    parser.add_argument('--rnn_type',                   type=str,   default=['gru', 'lstm'], nargs='+', help='RNN type (GRU or LSTM)')
    parser.add_argument('--num_rnn_layers',             type=int,   default=[1, 2, 4, 8], nargs='+', help='Number of RNN layers')
    parser.add_argument('--q_type',                     type=str,   default=['mu-law', 'linear'], nargs='+', help='Quantization type')
    parser.add_argument('--rnn_dropout',                type=float, default=[0.2, 0.4, 0.6], nargs='+', help='Size of the RNN dropout')
    parser.add_argument('--optimizer',                  type=str,   default='adam', choices=optimizer_factory.keys(),
                                                        help='Type of training optimizer to use')
    parser.add_argument('--learning_rate',              type=float, default=[1e-2, 1e-3, 1e-4], nargs='+', help='Learning rate of training')
    parser.add_argument('--momentum',                   type=float, default=[0.1, 0.5, 0.9], nargs='+', help='Optimizer momentum')
    return parser.parse_args()

args = get_arguments()

# We need to supply a list of frame sizes, but Keras Tuner doesn't have a list hp type. This
# workaround suggested at: https://github.com/keras-team/keras-tuner/issues/128#issuecomment-546013726
def AddChoice(hp, name, values):
    index = hp.Choice(name, list(range(len(values))))
    return values[index]

# Create and compile the model
def build_model(hp):
    model = SampleRNN(
        batch_size=hp.Choice('batch_size', args.batch_size),
        frame_sizes=AddChoice(hp, 'frame_sizes', args.frame_sizes),
        seq_len=hp.Choice('seq_len', args.seq_len),
        q_type=hp.Choice('q_type', args.q_type),
        q_levels=256,
        dim=hp.Choice('dim', args.dim),
        rnn_type=hp.Choice('rnn_type', args.rnn_type),
        num_rnn_layers=hp.Choice('num_rnn_layers', args.num_rnn_layers),
        emb_size=256,
        skip_conn=hp.Boolean('skip_conn'),
        rnn_dropout=hp.Choice('rnn_dropout', args.rnn_dropout)
    )
    optimizer = optimizer_factory[args.optimizer](
        learning_rate=hp.Choice('learning_rate', args.learning_rate),
        momentum=hp.Choice('momentum', args.momentum)
    )
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=optimizer, loss=compute_loss, metrics=[train_accuracy])
    return model

# Tuner subclass.
class SampleRNNTuner(kt.Tuner):

    def run_trial(self, trial, data_dir, val_frac, objective, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)

        num_epochs = kwargs.get('num_epochs')

        batch_size = model.batch_size
        seq_len = model.seq_len
        overlap = model.big_frame_size
        q_type = model.q_type
        q_levels = 256

        (train_split, val_split) = get_dataset_filenames_split(
            data_dir, val_frac, model.batch_size)

        # Train, Val and Test Datasets
        train_dataset = get_dataset(train_split, num_epochs, batch_size, seq_len, overlap,
                                    drop_remainder=True, q_type=q_type, q_levels=q_levels)
        val_dataset = get_dataset(val_split, 1, batch_size, seq_len, overlap, shuffle=False,
                                  drop_remainder=True, q_type=q_type, q_levels=q_levels)

        # Get subseqs per batch...
        samples0, _ = librosa.load(train_split[0], sr=None, mono=True)
        steps_per_batch = int(np.floor(len(samples0) / float(seq_len)))

        # Get subseqs per epoch...
        steps_per_epoch = len(train_split) // batch_size * steps_per_batch

        # Train...
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            shuffle=False,
            validation_data=val_dataset 
        )

        # See https://github.com/keras-team/keras-tuner/blob/master/kerastuner/engine/multi_execution_tuner.py#L95
        metrics = collections.defaultdict()
        for metric, epoch_values in history.history.items():
            if self.oracle.objective.direction == 'min':
                best_value = np.min(epoch_values)
            else:
                best_value = np.max(epoch_values)
            metrics[metric] = best_value

        oracle_metrics_dict = {objective: metrics[objective]}

        # If we completely override run_trial we need to call this at the end.
        # See https://keras-team.github.io/keras-tuner/documentation/tuners/#run_trial-method_1 
        self.oracle.update_trial(trial.trial_id, oracle_metrics_dict)
        self.save_model(trial.trial_id, model)


# Random Search.
def create_random_search_optimizer(objective=args.objective, max_trials=args.max_trials, seed=None):
    return SampleRNNTuner(
        oracle=kt.oracles.RandomSearch(
            objective=objective,
            max_trials=max_trials,
            seed=seed),
        hypermodel=build_model,
        directory='./logdir/tuner',
        project_name=args.id)

# Bayesian Optimization.
def create_bayesian_optimizer(objective=args.objective, max_trials=args.max_trials,
                              num_initial_points=None, alpha=0.0001, beta=2.6, seed=None):
    return SampleRNNTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=seed),
        hypermodel=build_model,
        directory='./logdir/tuner',
        project_name=args.id)

tuner_factory = {
    'bayesian' : create_bayesian_optimizer,
    'random_search' : create_random_search_optimizer
}

tuner = tuner_factory[args.type]()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor = args.objective,
        patience = args.early_stopping_patience)
]

tuner.search(
    data_dir=args.data_dir,
    val_frac=args.val_frac,
    objective=args.objective,
    num_epochs=args.num_epochs,
    callbacks=callbacks,
)

print('\n')
print('Printing search summary...')
tuner.results_summary()

# Run it like:
'''
nohup python keras_tuner.py \
  --data_dir path/to/dataset \
  --num_epochs 2 \
  --frame_sizes 16 64 \
  --frame_sizes 32 128 \
  --batch_size 16 32 64 \
  --seq_len 512 1024 2048 \
  --num_rnn_layers 2 4 \
  > tuner.log 2>&1 </dev/null &
'''