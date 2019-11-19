import os
import numpy as np
import tensorflow as tf
import time
import json
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile


def tf_f1_score():
    """f1 score using tf contrib metrics"""
    def f1_score(y_true, y_pred):
        score = tf.contrib.metrics.f1_score(y_true, predictions=K.clip(y_pred, 0, 1))[1]
        K.get_session().run(tf.local_variables_initializer())
        return score

    return f1_score


def tf_roc_auc():
    """ROC AUC metric using tf contrib metrics"""
    def roc_auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, predictions=K.clip(y_pred, 0, 1), curve='ROC')[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
    return roc_auc


def tf_pr_auc():
    """PR AUC metric using tf contrib metrics"""
    def pr_auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, predictions=K.clip(y_pred, 0, 1), curve='PR', summation_method='careful_interpolation')[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
    return pr_auc


class MetricsBoard(TensorBoard):
    def __init__(self,
                 log_dir='./logs',
                 update_freq='batch',
                 steps_per_log=1,
                 **kwargs):
        self.log_step = 0
        self.steps_per_log = steps_per_log
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.training_log_dir = os.path.join(log_dir, 'training')
        super(MetricsBoard, self).__init__(log_dir=self.training_log_dir, update_freq=update_freq, **kwargs)

    def set_model(self, model):
        """Set model params and params useful for writing"""
        self.model = model
        self.sess = K.get_session()
        self.train_file_writer = tf.summary.FileWriter(self.training_log_dir)
        self.val_file_writer = tf.summary.FileWriter(self.val_log_dir)
        super(MetricsBoard, self).set_model(model)

    def on_batch_end(self, batch, logs):
        """What to write at the end of each batch to tensorboard"""
        if self.log_step % self.steps_per_log == 0:
            self._write_logs(logs=logs, writer=self.train_file_writer, index=self.log_step, prefix='batch_')
        self.log_step += 1

    def on_epoch_end(self, epoch, logs):
        """What to write at the end of each epoch to tensorboard. Uses seperate filewriters for validation/training"""
        """Plot logs on same graph using seperate writers"""
        # logs.update({'learning_rate': float(K.get_value(self.model.optimizer.lr))})
        val_logs = {key.replace('val_', ''): val for key, val in logs.items() if key.startswith('val_')}
        train_logs = {key: val for key, val in logs.items() if not key.startswith('val_')}
        self._write_logs(logs=train_logs, writer=self.train_file_writer, index=epoch, prefix='epoch_')
        self._write_logs(logs=val_logs, writer=self.val_file_writer, index=epoch, prefix='epoch_')

    def _write_logs(self, logs, writer, index, prefix=''):
        """Helper method to write logs"""
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = prefix + name
            writer.add_summary(summary, index)
        writer.flush()

    def on_train_end(self, _):
        """Clean up"""
        self.train_file_writer.close()
        self.val_file_writer.close()


def get_model_evaluations(model, dataset, batch_size=300000):
    """Get model evaluations for a specific dataset -- batch size should
    be expected to be approximately the size of the dataset"""
    data_unbatched = dataset.apply(tf.data.experimental.unbatch())
    data_rebatched = data_unbatched.batch(batch_size)

    # evaluate test set
    eval_names = model.metrics_names
    eval_metrics = model.evaluate(data_rebatched, steps=1)
    eval_dict = dict(zip(eval_names, eval_metrics))

    # get probabilities and percentiles
    test_labels = data_rebatched.map(lambda x, y: y)
    test_labels_iterator = test_labels.make_initializable_iterator()
    K.get_session().run(test_labels_iterator.initializer)

    y_pred = tf.clip_by_value(model.predict(data_rebatched, steps=1), 0, 1)
    y_pred = K.get_session().run(y_pred).flatten()
    prob_names = ['p05', 'p10', 'p50', 'p75', 'p90', 'p95']
    p_tiles = [5, 10, 50, 75, 90, 95]
    prob_percents = np.percentile(y_pred, p_tiles)
    prob_dict = dict(zip(prob_names, prob_percents))

    # output dict
    output_dict = {}
    output_dict.update(eval_dict)
    output_dict.update(prob_dict)
    return output_dict, y_pred


def output_model_evaluations(model, test_dataset, batch_size, model_dir, train_val_history=None):
    """Generate output model evaluations and predictions"""

    def generate_model_eval_output(test_eval_dict):
        """Create output dictionary of metrics for testset, will do it for train/val if history is present"""
        eval_output_dict = dict()
        eval_output_dict['test'] = test_eval_dict

        if train_val_history is not None:
            training_output_dict = {key: val[-1] for key, val in train_val_history.items() if not key.startswith('val_')}
            validation_output_dict = {key.replace('val_', ''): val[-1] for key, val in train_val_history.items() if key.startswith('val_')}
            eval_output_dict['training'] = training_output_dict
            eval_output_dict['validation'] = validation_output_dict
        model_eval_loc = os.path.join(model_dir, "evaluations.json")
        with open(model_eval_loc, 'w') as outfile:
            json.dump(str(eval_output_dict), outfile, indent=4, sort_keys=True)
        return

    def generate_test_predictions(probabilities, model_dir):
        """Output testset predictions to a csv file"""
        _batch_limit = 100000
        if batch_size > _batch_limit:
            print("WARNING: outputting test predictions of batchsize=%d to csv file" % batch_size)
        predictions_loc = os.path.join(model_dir, "predictions.csv")
        with open(predictions_loc, 'w') as outfile:
            np.savetxt(outfile, probabilities, header="test_probabilities", comments='')
        return

    test_output_dict, test_y_pred_probs = get_model_evaluations(model=model,
                                                                dataset=test_dataset,
                                                                batch_size=batch_size)
    generate_model_eval_output(test_eval_dict=test_output_dict)
    generate_test_predictions(probabilities=test_y_pred_probs, model_dir=model_dir)
    return


def create_weighted_binary_crossentropy(zero_weight, one_weight):
    """
    weight cross entropy for imbalanced data set
    :param zero_weight:
    :param one_weight:
    :return:
    """
    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy


def batch_eval(model, test_dataset, file_name='./result.npy', max_size=1000000):
    """
    batch evaluation on test datset and save result to numpy file
    :param model:
    :param test_dataset:
    :param file_name:
    :param max_size:
    :return:
    """
    iter_test = test_dataset.make_one_shot_iterator()
    next_element = iter_test.get_next()

    result = np.array([])
    labels = np.array([])
    game_ids = np.array([])
    while True:
        try:
            if result.size > max_size:
                break
            x, y = K.get_session().run(next_element)
            p = model.predict(x)
            labels = np.append(labels, np.array(y).reshape(p.size))
            result = np.append(result, np.array(p).reshape(p.size))
            game_ids = np.append(game_ids, np.array(x['sourceGameId']).reshape(p.size))

        except tf.errors.OutOfRangeError:
            break

    pairs = np.stack((labels, result, game_ids), axis=-1)
    np.save(file_io.FileIO(file_name, 'wb'), np.array(pairs), allow_pickle=False)
    print("DONE!")


def make_batched_features_dataset(  file_pattern,
                                    batch_size,
                                    features,
                                    reader=core_readers.TFRecordDataset,
                                    label_key=None,
                                    weight_key=None,
                                    reader_args=None,
                                    num_epochs=None,
                                    shuffle=True,
                                    shuffle_buffer_size=10000,
                                    shuffle_seed=None,
                                    prefetch_buffer_size=optimization.AUTOTUNE,
                                    reader_num_threads=32,
                                    parser_num_threads=32,
                                    sloppy_ordering=True,
                                    drop_final_batch=False):

    """Returns a `Dataset` of feature dictionaries from `Example` protos.
    Returns:
    A dataset of `dict` elements, (or a tuple of `dict` elements and label).
    Each `dict` maps feature keys to `Tensor` or `SparseTensor` objects.
    """
    if shuffle_seed is None:
        shuffle_seed = int(time.time())

    filenames = list(gfile.Glob(file_pattern))
    dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.shuffle(len(filenames), shuffle_seed)

    # Read `Example` records from files as tensor objects.
    if reader_args is None:
        reader_args = []

    # Read files sequentially (if reader_num_threads=1) or in parallel
    dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          block_length=200,
          sloppy=sloppy_ordering))

    # Extract values if the `Example` tensors are stored as key-value tuples.
    if dataset_ops.get_legacy_output_types(dataset) == (
          dtypes.string, dtypes.string):
        dataset = dataset_ops.MapDataset(
          dataset, lambda _, v: v, use_inter_op_parallelism=True)

    # Apply dataset repeat and shuffle transformations.
    dataset = dataset.apply(
        shuffle_ops.shuffle_and_repeat(shuffle_buffer_size, num_epochs,
                                       shuffle_seed))

    dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

    # Parse `Example` tensors to a dictionary of `Feature` tensors.
    dataset = dataset.apply(
      parsing_ops.parse_example_dataset(
          features, num_parallel_calls=parser_num_threads))

        
    if weight_key:
        #assert label_key
        #assert label_key != weight_key
        #assert label_key in features
        assert weight_key in features
        if label_key:
            if label_key not in features:
                raise ValueError(
                    "The 'label_key' provided (%r) must be one of the 'features' keys."% label_key)
        assert label_key != weight_key
        
        
        dataset = dataset.map(lambda x: (x, x.pop(label_key),x.pop(weight_key)))
        #w = dataset.map(lambda x,y : x.pop(weight_key))
        
    else:
        if label_key:
            if label_key not in features:
                raise ValueError(
                    "The `label_key` provided (%r) must be one of the `features` keys." % label_key)
        dataset = dataset.map(lambda x: (x, x.pop(label_key)))
    dataset = dataset.prefetch(prefetch_buffer_size)
    
    if not weight_key:
        return dataset
    else:
        return dataset

def make_batched_features_dataset_multi_task(  file_pattern,
                                    batch_size,
                                    features,
                                    reader=core_readers.TFRecordDataset,
                                    label_key=None,
                                    weight_key=None,
                                    reader_args=None,
                                    num_epochs=None,
                                    shuffle=True,
                                    shuffle_buffer_size=10000,
                                    shuffle_seed=None,
                                    prefetch_buffer_size=optimization.AUTOTUNE,
                                    reader_num_threads=32,
                                    parser_num_threads=32,
                                    sloppy_ordering=True,
                                    drop_final_batch=False):

    """Returns a `Dataset` of feature dictionaries from `Example` protos.
    Returns:
    A dataset of `dict` elements, (or a tuple of `dict` elements and label).
    Each `dict` maps feature keys to `Tensor` or `SparseTensor` objects.
    """
    if shuffle_seed is None:
        shuffle_seed = int(time.time())

    filenames = list(gfile.Glob(file_pattern))
    dataset = dataset_ops.Dataset.from_tensor_slices(filenames)
    if shuffle:
        dataset = dataset.shuffle(len(filenames), shuffle_seed)

    # Read `Example` records from files as tensor objects.
    if reader_args is None:
        reader_args = []

    # Read files sequentially (if reader_num_threads=1) or in parallel
    dataset = dataset.apply(
      interleave_ops.parallel_interleave(
          lambda filename: reader(filename, *reader_args),
          cycle_length=reader_num_threads,
          block_length=200,
          sloppy=sloppy_ordering))

    # Extract values if the `Example` tensors are stored as key-value tuples.
    if dataset_ops.get_legacy_output_types(dataset) == (
          dtypes.string, dtypes.string):
        dataset = dataset_ops.MapDataset(
          dataset, lambda _, v: v, use_inter_op_parallelism=True)

    # Apply dataset repeat and shuffle transformations.
    dataset = dataset.apply(
        shuffle_ops.shuffle_and_repeat(shuffle_buffer_size, num_epochs,
                                       shuffle_seed))

    dataset = dataset.batch(
      batch_size, drop_remainder=drop_final_batch or num_epochs is None)

    # Parse `Example` tensors to a dictionary of `Feature` tensors.
    dataset = dataset.apply(
      parsing_ops.parse_example_dataset(
          features, num_parallel_calls=parser_num_threads))

        
    if weight_key:
        #assert label_key
        #assert label_key != weight_key
        #assert label_key in features
        assert weight_key in features
        if label_key:
            if label_key not in features:
                raise ValueError(
                    "The 'label_key' provided (%r) must be one of the 'features' keys."% label_key)
        assert label_key != weight_key
        
        
        dataset = dataset.map(lambda x: (x, tuple([x.pop(label_key)]*5),x.pop(weight_key)))
        #w = dataset.map(lambda x,y : x.pop(weight_key))
        
    else:
        if label_key:
            if label_key not in features:
                raise ValueError(
                    "The `label_key` provided (%r) must be one of the `features` keys." % label_key)
        dataset = dataset.map(lambda x: (x, tuple([x.pop(label_key)]*5)))
    dataset = dataset.prefetch(prefetch_buffer_size)
    
    if not weight_key:
        return dataset
    else:
        return dataset
   