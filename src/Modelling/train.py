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
    
    def load_my_model(self):
        """Function that loads a past model.
        Args:
            path (str): path to the wanted model, 
            custom_objects (dict): dictionnary of the custom object integrated into the model

        Returns:
            loaded_model (keras.engine.sequential.Sequential)
        """
        path = self.conf["paths"]["Outputs_path"] + self.conf["paths"]["folder_model"]
        model = self.conf["model"]["model_name"]
        custom_objects = {'KerasLayer':hub.KerasLayer,
                          'macro_f1': self.macro_f1,
                          'macro_soft_f1':self.macro_soft_f1}
        loaded_model = keras.models.load_model(path + model,
                                               custom_objects)

        return loaded_model


