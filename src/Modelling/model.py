"""Create Model Class."""

import logging
import warnings

import tensorflow as tf
import tensorflow_hub as hub

import keras
from tensorflow.keras import layers


warnings.filterwarnings('ignore')
logger = logging.getLogger('main_logger')

class Model():
    """Create the model class with model architecture
    and metrics for training.
    """

    def __init__(self, conf):
        self.conf = conf

    @tf.function
    def macro_soft_f1(self, y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        # reduce 1 - soft-f1 in order to increase soft-f1
        cost = 1 - soft_f1
        # average on all labels
        macro_cost = tf.reduce_mean(cost)
        return macro_cost

    @tf.function
    def macro_f1(self, y, y_hat):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)

        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive

        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        thresh = self.conf["model"]["THRESH"]
        y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1

    def create_model(self):
        """Creates a model composed of a pretrained model from the
        tensorflow hub, regularization layers (dropout, batch normalization)
        and dense layers.

        Args:
            conf: model config paramaters.

        Returns:
            model (keras.engine.sequential.Sequential)
        """
        logger.info("Creating the model...")
        conf_model = self.conf["model"]
        feature_extractor_url = conf_model["FEATURE_EXTRACTOR_URL"]
        img_size = conf_model["IMG_SIZE"]
        channels = conf_model["CHANNELS"]
        N_LABELS = conf_model["N_LABELS"]
        lr = conf_model["LR"]
        feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                                 input_shape=(img_size, img_size, channels))

        feature_extractor_layer.trainable = False

        model = tf.keras.Sequential([
            feature_extractor_layer,
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu', name='hidden_layer_1'),
            layers.Dense(256, activation='relu', name='hidden_layer_2'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu', name='hidden_layer_3'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', name='hidden_layer_4'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', name='hidden_layer_5'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(N_LABELS, activation='sigmoid', name='output')
        ])

        model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
          loss=self.macro_soft_f1,
          metrics=[self.macro_f1])

        return model


