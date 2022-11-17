"""Training of the model."""

import logging
import warnings
import os
import tensorflow as tf
from time import time
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger('main_logger')


class Train():
    """Create a class to train and evaluate the model"""
    
    def __init__(self, conf, model=None, train_ds=None, test_ds=None):
        self.conf = conf
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds

    def training(self):
        """Train a deep learning model based on Keras tensorflow

        Args:
            model (keras.engine.sequential.Sequential)
            train_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset), 
            test_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset), 
            conf: model configuration parameters

        Returns:
            history: history callback of fitting a tensorflow keras model
        """
        path_out = self.conf["paths"]["Outputs_path"] + self.conf["paths"]["folder_model"]
        epochs = self.conf["model"]["EPOCHS"]
        now = datetime.now()
        timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
        model_name = "model" + timestamp
        start = time()
        history = self.model.fit(self.train_ds,
                                 epochs=epochs,
                                 validation_data=self.test_ds,
                                 callbacks=[
                                     tf.keras.callbacks.ModelCheckpoint(
                                         os.path.join(path_out, model_name),
                                         monitor='val_loss', save_best_only=True
                                     )
                                 ]
                                )
        logging.info(f'Training took {round(time()-start, 2)} sec')
        return history


