#%%writefile promo_trainer.py
import sys; sys.argv = ['']        
import argparse
from promo_model import WDModel
import trainer_utils

from trainer_utils import box, get_dataset

import tensorflow as tf
import tensorflow_transform as tft
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")



def train(model,
          dataset_train,
          steps_per_epoch, 
          validation_data , 
          validation_steps,
          epochs,
          callbacks):
    #box ('Model {} is training'.format(model.name))
    model.fit(dataset_train,
              steps_per_epoch=steps_per_epoch, 
              validation_data = validation_data, 
              validation_steps=validation_steps,
              epochs=epochs,
              callbacks=callbacks,
              verbose=0)
    
    #box ('Model {} training is finished.'.format(model.name), symb='+')
    
    return model
    
def main(dataset_train,dataset_test):
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--use_weight',
        help = 'Wether to use weights or not',
        type = bool,
        default = 0
    )
    
    parser.add_argument(
        '--data_dir',
        help = 'Tft schema',
        type = str,
        default = "gs://unity-ads-ds-prd-users/villew/promo/output4/"
    )
    
    
    parser.add_argument(
        '--log_dir',
        help = 'log directory',
        type = str,
        default = "train"
    )    
    
    parser.add_argument(
        '--run_id',
        help = 'Name/id for run',
        type = str,
        default = "Testing-0"
    )
    
    parser.add_argument(
        '--lr',
        help = 'Learning Rate',
        type = float,
        default = 10**-2
    )
    
    args = parser.parse_args()
    #model = WDModel('test',args).model
    model = WDModel(args).model
    
    tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=args.log_dir),
                            trainer_utils.MetricsBoard()]
    
    '''
    # Assign model variables to commandline arguments
    model.TRAIN_PATHS = args.train_data_paths
    model.BATCH_SIZE = args.batch_size
    model.HIDDEN_UNITS = args.hidden_units
    model.OUTPUT_DIR = args.output_dir
    # Run the training job
    model.train_and_evaluate()
    
    '''

    
    #box ('Model success' if len(model.layers)>0 else 'Model failed!',symb='+')
    model = train(model,
                  dataset_train,
                  steps_per_epoch=646, 
                  validation_data = dataset_test, 
                  validation_steps=683,
                  epochs=5,
                  callbacks=tensorboard_callback)
    
    
if __name__ == '__main__':
    tf_transform_output = tft.TFTransformOutput("gs://unity-ads-ds-prd-users/villew/promo/output4/")
    use_weight = 0

    weight_key = "campaignCost_mod"
    weight_key = weight_key if use_weight else None

    batch_size = 1 * 10 ** 4

    TRAIN_BUCKET_PATH = "gs://unity-ads-ds-prd-users/villew/promo/output4/train_tfrecord-00000-of-00001"
    TEST_BUCKET_PATH = "gs://unity-ads-ds-prd-users/villew/promo/output4/test_tfrecord-00000-of-00001"
    
    label_key="label"
    weight_key=weight_key
    
    
    dataset_train = get_dataset(TRAIN_BUCKET_PATH, tf_transform_output, label_key, weight_key, batch_size)
    dataset_test = get_dataset(TEST_BUCKET_PATH, tf_transform_output, label_key, weight_key, batch_size)
    
    #box ('Weights are used!' if len(next(iter(dataset_train)))==3 else 'No weights!')
    
    main(dataset_train,dataset_test)
