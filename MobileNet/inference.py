import logging
import os
import warnings

import numpy as np
import pandas as pd

from keras.preprocessing import image
import keras

import tensorflow_hub as hub

from tools.model import f1_class

from config import *

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def load_my_model(path=PATH_TO_MODEL_TO_LOAD, custom_objects={'KerasLayer':hub.KerasLayer, 'macro_f1':f1_class().macro_f1, 'macro_soft_f1':f1_class().macro_soft_f1}):
    """Function that loads a past model.
    Args:
        path (str): path to the wanted model, 
        custom_objects (dict): dictionnary of the custom object integrated into the model
    
    Returns:
        loaded_model (keras.engine.sequential.Sequential)
    """    
    loaded_model = keras.models.load_model(path,
                                           custom_objects)
    
    return loaded_model


def make_prediction(IDs, model, submission_name=None, path_to_saved_submissions_folder=PATH_TO_SAVED_SUBMISSIONS_FOLDER, testing_path=TESTING_PATH, thresh=PREDICTION_THRESH, save=True):
    """Function that returns a dataframe with the predictions.
    Args:
        IDs (list): list of the ids of the tested images, 
        model (keras.engine.sequential.Sequential), 
        submission_name (None or str): name of the csv in case the user wants to save the predictions, 
        path_to_saved_submissions_folder (str): path to the folder where the csv will be saved, 
        testing_path (str): path to the tested images, 
        thresh (float): probability value above which we predict a positive value, 
        save (Boolean): indicates if the user wants to save as csv the predictions
    
    Returns:
        df (Pandas DataFrame): predictions
    """
    if save==True:
        assert type(submission_name) == str
        
    for idx, ID in enumerate(IDs):
        img_path = os.path.join(testing_path, str(ID))

        # Read and prepare image
        img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
        img = image.img_to_array(img)
        img = img/255
        img = np.expand_dims(img, axis=0)

        # Generate prediction
        prediction = (model.predict(img) > thresh).astype('int')[0]
        if idx == 0:
            predictions = prediction
        else:
            predictions = np.vstack([predictions, prediction])
    
    template_test = pd.read_csv(TEMPLATE_TEST_PATH)
    df = pd.DataFrame(predictions, columns=template_test.columns[1:])
    df['filename'] = template_test.filename
    df = df[template_test.columns]
    
    if save==True:
        df.to_csv(path_to_saved_submissions_folder + submission_name, index=False)
    
    return df
