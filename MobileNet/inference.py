import logging
import os
import warnings

import numpy as np
import pandas as pd

from keras.preprocessing import image

from config import *

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


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

    df = pd.DataFrame(predictions, columns=template_test.columns[1:])
    df['filename'] = template_test.filename
    df = df[['filename', 
             'FISSURE', 
             'REPARATION', 
             'FISSURE LONGITUDINALE', 
             'MISE EN DALLE']]
    
    if save==True:
        # assert type(submission_name) == str
        df.to_csv(path_to_saved_submissions_folder + submission_name, index=False)
    
    return df