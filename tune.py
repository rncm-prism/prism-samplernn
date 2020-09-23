import tensorflow as tf
import kerastuner as kt
import json
import random

from samplernn import SampleRNN
#from dataset import (get_dataset, get_dataset_filenames_split)
from dataset import (get_dataset, find_files)
from train import optimizer_factory


# Create and compile the model
def build_model(hp):
    hp.Choice('big_frame_size', [32, 64, 128], default=64)
    hp.Choice('frame_size', [4, 2])
    model = SampleRNN(
        batch_size=hp.Choice('batch_size', [16, 32, 64, 128], default=32),
        frame_sizes=[
            hp['big_frame_size'] // hp['frame_size'],
            hp['big_frame_size']
        ],
        seq_len=hp.Choice('seq_len', [512, 1024, 2048], default=1024),
        q_type='mu-law',
        q_levels=256,
        dim=hp.Choice('dim', [1024, 2048], default=1024),
        rnn_type='gru',
        num_rnn_layers=hp.Choice('num_rnn_layers', [1, 2, 4, 8], default=4),
        emb_size=256,
        skip_conn=False
    )
    optimizer = optimizer_factory['adam'](
        learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3),
        momentum=0.9,
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model

def get_dataset_filenames_split(data_dir, val_pcnt, test_pcnt):
    files = find_files(data_dir)
    if not files:
        raise ValueError("No audio files found in '{}'.".format(data_dir))
    random.shuffle(files)
    num_files = len(files)
    test_start = int( (1 - test_pcnt) * num_files )
    val_start = int( (1 - test_pcnt - val_pcnt) * num_files )
    return files[: val_start], files[val_start : test_start], files[test_start :]


# Tuner subclass.
class SampleRNNTuner(kt.Tuner):

    def run_trial(self, trial, data_dir, val_pcnt, test_pcnt, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(trial.hyperparameters)

        num_epochs = kwargs.get('num_epochs')

        (train_split, val_split, test_split) = get_dataset_filenames_split(data_dir, val_pcnt, test_pcnt)

        batch_size = model.batch_size
        seq_len = model.seq_len
        overlap = model.big_frame_size
        q_type = 'mu-law'
        q_levels = 256

        # Train, Val and Test Datasets
        train_dataset = get_dataset(train_split, num_epochs, batch_size, seq_len, overlap,
                                    drop_remainder=True, q_type=q_type, q_levels=q_levels)
        val_dataset = get_dataset(val_split, 1, batch_size, seq_len, overlap,
                                  drop_remainder=True, q_type=q_type, q_levels=q_levels)
        test_dataset = get_dataset(test_split, 1, batch_size, seq_len, overlap,
                                   drop_remainder=True, q_type=q_type, q_levels=q_levels)

        # Train...
        model.fit(
            train_dataset,
            epochs=num_epochs,
            shuffle=False,
            validation_data=val_dataset 
        )

        # Evaluate...
        (val_loss, _) = model.evaluate(
            test_dataset,
            steps=10,
            verbose=0
        )

        # If we completely override run_trial we need to call this at the end.
        # See https://keras-team.github.io/keras-tuner/documentation/tuners/#run_trial-method_1 
        self.oracle.update_trial(trial.trial_id, {'loss': val_loss})
        self.oracle.save_model(trial.trial_id, model)


# Random Search.
def create_random_search_optimizer(objective='loss', max_trials=2, seed=None):
    return SampleRNNTuner(
        oracle=kt.oracles.RandomSearch(
            objective=objective,
            max_trials=max_trials,
            seed=seed),
        hypermodel=build_model)

# Bayesian Optimization.
def create_bayesian_optimizer(objective='loss', max_trials=2, num_initial_points=None,
                              alpha=0.0001, beta=2.6, seed=None):
    return SampleRNNTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=seed),
        hypermodel=build_model)

# Run it like -
# tuner = create_bayesian_optimizer()
# tuner.search('./path/to/dataset', 0.1, 0.1, num_epochs=5)