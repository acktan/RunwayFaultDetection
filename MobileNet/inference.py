import logging
import os
import warnings

import numpy as np
import pandas as pd

from keras.preprocessing import image

from config import *

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def make_prediction(IDs, model, submission_name, path_to_saved_submissions_folder=PATH_TO_SAVED_SUBMISSIONS_FOLDER, testing_path=TESTING_PATH, thresh=PREDICTION_THRESH, save=True):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS, 
        channels (int): number of channel, 3 by default when rgb,
        img_size (int): image height and width,
        max_delta (float), 
        seed (tensor): a shape [2] Tensor, the seed to the random number generator. Must have dtype int32 or int64.
    
    Returns:
        image_augmented (tensorflow.python.data.ops.dataset_ops.ParallelMapDataset)
        label (numpy.ndarray): array of labels
    """
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
        df.to_csv(path_to_saved_submissions_folder + submission_name, index=False)
    
    return df