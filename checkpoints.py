from __future__ import print_function
import time

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from generate import generate


ERASE_LINE = '\x1b[2K'

# Custom training step callback (prints training stats).
class TrainingStepCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, num_epochs, steps_per_epoch, steps_per_batch,
                 resume_from, verbose=True):
        self.model = model
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_batch = steps_per_batch
        self.resume_from = resume_from
        self.verbose = verbose
        self.epoch_start_time = 0
        self.step_start_time = 0

    def on_train_begin(self, logs):
        if self.resume_from:
            self.model.load_weights(self.resume_from)

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch + 1
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs):
        loss, acc = logs.get('loss'), logs.get('accuracy')
        val_loss, val_acc = logs.get('val_loss'), logs.get('val_accuracy')
        step = self.steps_per_epoch
        self._print_step_stats(step, loss, acc, val_loss, val_acc)

    def on_train_batch_begin(self, batch, logs):
        if batch % self.steps_per_batch == 0 : self.model.reset_states()
        self.step_start_time = time.time()

    def on_train_batch_end(self, batch, logs):
        loss, acc = logs.get('loss'), logs.get('accuracy')
        step = batch + 1
        self._print_step_stats(step, loss, acc)

    def on_test_batch_end(self, batch, logs):
        self.on_batch_end(batch, logs)
    
    # Print the stats for one training step...
    def _print_step_stats(self, step, loss, acc, val_loss=None, val_acc=None):
        epoch_string = f'Epoch: {self.epoch}/{self.num_epochs}'
        step_string = f'Total Steps: {self.steps_per_epoch}' if val_loss else f'Step: {step}/{self.steps_per_epoch}'
        val_string = f'Val Loss: {val_loss:.3f}, Val Accuracy: {val_acc*100:.3f}, ' if val_loss else ""
        dur_string = format_epoch_dur(time.time()-self.epoch_start_time) if val_loss \
            else f'{time.time()-self.step_start_time:.3f} sec/step'
        end_char = '\r' if (self.verbose == False) and (step != self.steps_per_epoch) else '\n'
        stats_string = f'{epoch_string}, {step_string}, Loss: {loss:.3f}, Accuracy: {acc*100:.3f}, {val_string}({dur_string})'
        if self.verbose == False : stats_string = ERASE_LINE + stats_string
        print(stats_string, end=end_char)

def format_epoch_dur(secs):
    mins = int(secs // 60)
    hrs = int(mins // 60)
    sec_str = f'{int(secs) % 60}' if float(secs).is_integer() else f'{secs % 60:.3f}'
    if hrs > 0:
        return f'{hrs} hrs {mins % 60} min {sec_str} sec'
    elif mins > 0:
        return f'{mins} min {sec_str} sec'
    else:
        return f'{sec_str} seconds'

# Custom checkpoint callback. Manages generation phase and also
# deletes old checkpoints.
class ModelCheckpointCallback(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, dir, max_to_keep, generate, generation_args, *args, **kwargs):
        super(ModelCheckpointCallback, self).__init__(*args, **kwargs)
        self.dir = dir
        self.max_to_keep = max_to_keep
        self.generate = generate
        self.generation_args = generation_args
        self._maybe_delete = []

    def on_epoch_begin(self, epoch, logs):
        super(ModelCheckpointCallback, self).on_epoch_begin(epoch, logs)
        # Track the current epoch (1-indexed).
        self.epoch = epoch + 1
        # Track the last saved checkpoint (initially None).
        self.last_saved = tf.train.latest_checkpoint(self.dir)

    def on_epoch_end(self, epoch, logs=None):
        super(ModelCheckpointCallback, self).on_epoch_end(epoch, logs)
        ckpt_path = tf.train.latest_checkpoint(self.dir)
        # Track saved checkpoints, and remove old ones...
        if self.max_to_keep:
            if ckpt_path and ckpt_path not in self._maybe_delete:
                self._maybe_delete.append(ckpt_path)
            self._sweep()
        # Generate audio...
        if self.generate==True and (ckpt_path != self.last_saved):
            self._generate(ckpt_path, self.generation_args)

    def _generate(self, ckpt_path, args):
        print('Generating samples for epoch {}...'.format(self.epoch))
        output_path = '{}/{}_epoch_{}.wav'.format(args['generate_dir'], args['id'], self.epoch)
        generate(output_path, ckpt_path, args['config'], args['num_seqs'], args['dur'], args['sample_rate'],
                 args['temperature'], args['seed'], args['seed_offset'])

    def _delete_file_if_exists(self, filespec):
        """Deletes files matching `filespec`."""
        for pathname in file_io.get_matching_files(filespec):
            file_io.delete_file(pathname)

    # Adapted from the CheckpointManager class.
    def _sweep(self):
        """Deletes or preserves managed checkpoints."""
        while len(self._maybe_delete) > self.max_to_keep:
            filename = self._maybe_delete.pop(0)
            self._delete_file_if_exists(filename + ".index")
            self._delete_file_if_exists(filename + ".data-?????-of-?????")