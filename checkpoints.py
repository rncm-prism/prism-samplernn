from __future__ import print_function
import time

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from generate import generate


ERASE_LINE = '\x1b[2K'

# Custom training step callback (prints training stats and
# manages audio generation)
class TrainingStepCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, num_epochs, steps_per_epoch, steps_per_batch,
                 resume_from, verbose=True):
        self.model = model
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_batch = steps_per_batch
        self.resume_from = resume_from
        self.verbose = verbose
        self.step_start_time = 0

    def on_train_begin(self, logs):
        if self.resume_from:
            self.model.load_weights(self.resume_from)

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch + 1

    def on_batch_begin(self, batch, logs):
        if batch % self.steps_per_batch == 0 : self.model.reset_states()
        self.step_start_time = time.time()

    def on_batch_end(self, batch, logs):
        loss, acc = logs.get('loss'), logs.get('accuracy')
        step = batch + 1
        self._print_step_stats(step, loss, acc)

    def on_test_batch_end(self, batch, logs):
        self.on_batch_end(self, batch, logs):
    
    # Print the stats for one training step...
    def _print_step_stats(self, step, loss, acc):
        step_duration = time.time() - self.step_start_time
        template = 'Epoch: {:d}/{:d}, Step: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f}, ({:.3f} sec/step)'
        end_char = '\r' if (self.verbose == False) and (step != self.steps_per_epoch) else '\n'
        stats_string = template.format(self.epoch, self.num_epochs, step, self.steps_per_epoch, loss, acc * 100, step_duration)
        if self.verbose == False : stats_string = ERASE_LINE + stats_string
        print(stats_string, end=end_char)


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