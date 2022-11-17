"""Create inference class to make predictions."""

import logging
import os
import warnings

import numpy as np
import pandas as pd

from keras.preprocessing import image

warnings.filterwarnings('ignore')
logger = logging.getLogger('main_logger')


class Inference():
    """Create class inference to make predictions."""

    def __init__(self, conf, model, submission_name=None):
        self.conf = conf
        self.submission_name = submission_name
        self.model = model

    def make_prediction(self):
        """Function that returns a dataframe with the predictions.
        Args:
            conf: path and model parameters
            model: model used for predictions
            submission_name: name of the saved submission

        Returns:
            df (Pandas DataFrame): predictions
        """
        logger.info("Making predictions...")
        save = self.conf["model"]["save_preds"]
        testing_path = self.conf["paths"]["model_input_path"] + self.conf["paths"]["folder_test"]
        test_label_path = self.conf["paths"]["model_input_path"] + self.conf["paths"]["test_label_file"]
        IMG_SIZE = self.conf["model"]["IMG_SIZE"]
        CHANNELS = self.conf["model"]["CHANNELS"]
        thresh = self.conf["model"]["PREDICTION_THRESH"]
        test_labels = pd.read_csv(test_label_path)
        IDs = test_labels["filename"]

        if save==True:
            assert type(self.submission_name) == str

        for idx, ID in enumerate(IDs):
            img_path = os.path.join(testing_path, str(ID))

            # Read and prepare image
            img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
            img = image.img_to_array(img)
            img = img/255
            img = np.expand_dims(img, axis=0)

            # Generate prediction
            prediction = (self.model.predict(img) > thresh).astype('int')[0]
            if idx == 0:
                predictions = prediction
            else:
                predictions = np.vstack([predictions, prediction])

        df = pd.DataFrame(predictions, columns=test_labels.columns[1:])
        df['filename'] = test_labels.filename
        df = df[['filename', 
                 'FISSURE', 
                 'REPARATION', 
                 'FISSURE LONGITUDINALE', 
                 'MISE EN DALLE']]
        
        path = self.conf["paths"]["Outputs_path"] + self.conf["paths"]["folder_inference"]
        if save==True:
            logger.info(f"Saving model to {path+self.submission_name}.csv")
            df.to_csv(path + self.submission_name + ".csv", index=False)

        return df
