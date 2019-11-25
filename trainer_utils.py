%%writefile trainer_utils.py

from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
import tensorflow_transform as tft


import os,sys
import numpy as np
import tensorflow as tf
import time
import json

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import Callback
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
        self.train_file_writer = tf.compat.v1.summary.FileWriter(self.training_log_dir)
        self.val_file_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)
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
            summary = tf.compat.v1.Summary()
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
        
class OneHot(Layer):
    def __init__(self, num_class, **kwargs):
        super(OneHot, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_class = num_class

    def call(self, inputs, training=None):
        return K.one_hot(indices=tf.dtypes.cast(inputs, tf.int32), num_classes=self.num_class)

    def get_config(self):
        config = {'num_class': self.num_class}
        base_config = super(OneHot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    
def batch_eval(model, test_dataset, max_size=1000000):
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
    return np.array(pairs)



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

def get_dataset(tfrec_path, tf_transform_output, label_key, weight_key, batch_size):
    """Get dataset for model training
    Args:
      tfrec_path: Path of tensorflow records
      tf_transform_output: A TFTransform object
      feature_config: A list of FeatureColumns
      batch_size: batch size
    """
    dataset = make_batched_features_dataset(
        file_pattern=tfrec_path,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        label_key=label_key,
        weight_key=weight_key,
        shuffle=True)

    return dataset

    
def box(text, maxlen=59, symb='#'):
    n = len(text)
    str_len = int((maxlen-2-n)*1.0/2)
    
    str_len = int((maxlen-2-n)*1.0/2) if n>13 else int((maxlen-2-13)*1.0/2)
    print_statement = '''
    {}
    {} {:^13} {}
    {}
    '''.format(symb*maxlen,symb*str_len,text.strip().upper(),symb*str_len,symb*maxlen)
    print(print_statement)
    

class Metrics(Callback):
    def on_batch_end(self, batch, logs):
        return
    def on_epoch_end(self, epoch, logs):
        print ('epoch %d:' %(epoch))
        print ('loss=%f' %(logs['loss']))
        print ('auc-pr=%f' %(logs['auc-pr']))
        print ('auc-roc=%f' %(logs['auc-roc']))       
        return
