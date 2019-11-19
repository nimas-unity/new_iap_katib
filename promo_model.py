
import tensorflow as tf
import tensorflow_transform as tft
import train_utils
##########################
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Flatten, Input, Embedding, Reshape, Dense, \
    BatchNormalization, LayerNormalization, Dropout, GaussianNoise, ReLU, ELU, PReLU, Lambda

from tensorflow.keras.metrics import AUC, BinaryAccuracy, Recall, Precision
from tensorflow.keras.models import Model

from tensorflow.contrib.layers import sparse_column_with_keys,crossed_column
##########################
import numpy as np
import pandas as pd
import itertools
from trainer_utils import OneHot
from trainer_utils import batch_eval
import argparse

class WDModel():
    def __init__(self, args):
        self.model = None
        self.args = args
        
        inputs = []
        noise_layers = []
        numeric_layers = []
        embedding_layers = []
        onehot_layers = []
        
        #Numeric
        if not args.use_weight:
            feat_lst = ["campaignCost_mod", "startCountTotal", "purchaseCountTotal", "globalPurchaseCountTotal",
                        "globalStartCountTotal"]
        else:
            feat_lst = ["startCountTotal", "purchaseCountTotal", "globalPurchaseCountTotal",
                        "globalStartCountTotal"]

        for feat_key in feat_lst:
            input_layer = Input(shape=(1,), dtype='float32', name=feat_key)
            inputs.append(input_layer)
            numeric_layers.append(input_layer)   
            
            
        #Embedding
        tf_transform_output = tft.TFTransformOutput(args.data_dir)
        for feat_key in ["country", "game_campaignId", "game_zone", "platform", "sourceGameId"]:
            input_layer = Input(shape=(1,), dtype='int64', name=feat_key)
            inputs.append(input_layer)
            vocab_size = tf_transform_output.vocabulary_size_by_name(feat_key) + 1
            dim = 4 * int(vocab_size ** 0.25)
            embedding_layers.append(
                Embedding(vocab_size, dim, name="embedding_" + feat_key, embeddings_initializer='uniform')(input_layer)
            )

            
        #One-hot
        for feat_key in ["country", "game_campaignId", "game_zone", "platform", "sourceGameId"]:
            onehot_layer = OneHot(num_class=vocab_size)(input_layer)
            onehot_layers.append(
                Flatten(name='flatten_one_hot_{}'.format(feat_key))(onehot_layer)
            )  
        
        
        #Wide Model
        concat_onehot = tf.keras.layers.concatenate(onehot_layers, name='concat_onehot')
        wide_preds = Dense(1, name='wide_output', kernel_initializer='he_normal')(concat_onehot)
        wide_model = Model(inputs=inputs, outputs=wide_preds)
        
        

        
        #Deep Model
        concat_numeric = tf.keras.layers.concatenate(numeric_layers, name='concat_numeric')
        #Noise
        concat_numeric = GaussianNoise(stddev=0.03)(concat_numeric)
        
        concat_embedded = tf.keras.layers.concatenate(embedding_layers, name='concat_embedded')
        concat_embedded = Flatten()(concat_embedded)
        concat_all = tf.keras.layers.concatenate([concat_numeric, concat_embedded], name='concat_all')

        hidden1 = Dense(128, activation='selu', name='hidden1', kernel_initializer='he_normal')(concat_all)
        hidden1 = BatchNormalization(name='BN_hidden1')(hidden1)
        hidden2 = Dense(64, activation='selu', name='hidden2', kernel_initializer='he_normal')(hidden1)
        hidden2 = BatchNormalization(name='BN_hidden2')(hidden2)
        deep_preds = Dense(1,  name='deep_output', kernel_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.05,l2=0.05))(hidden2)
        deep_model = Model(inputs=inputs, outputs=deep_preds)


        #Wide-Deep Model
        merged_out = tf.keras.layers.concatenate([wide_model.output, deep_model.output])
        merged_out = Dense(1, activation='sigmoid')(merged_out)
        model = Model(inputs=inputs, outputs=merged_out, name=args.run_id)
        
        
        metrics = [AUC(num_thresholds=500, curve='PR', name='auc_pr'),
                   AUC(num_thresholds=500, curve='ROC', name='auc-roc')]

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        ##################
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=metrics)

        self.model = model
        
        
        
