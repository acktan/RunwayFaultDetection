"""Unit tests for model functions. It concerns the dataset.py, model.py, train.py, main.py and inference.py files"""

import pandas as pd
import numpy as np

import re
import os

from tools.train import *
from tools.dataset import *
from tools.model import *
from inference import *
import main

from config import *
from test_config import *

def test_create_dataset():
    """
    Test that the dataset are well created and according to the parameters we specified.
    """
    labels_train = pd.read_csv(LABEL_TRAIN_PATH)

    X = [os.path.join(DATASET_PATH, str(f)) for f in labels_train['filename']]
    y = labels_train.drop('filename', axis=1).values
    
    train_ds = create_dataset(filenames=X, 
                              labels=y, 
                              batch_size=BATCH_SIZE, 
                              is_training=True)
    
    for f, l in train_ds.take(1):
        assert f.numpy().shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)
        assert l.numpy().shape == (BATCH_SIZE, N_LABELS)


def test_make_prediction():
    """
    Test that the predictions are of the right shape.
    """
    model = load_my_model()
    template_test = pd.read_csv(TEMPLATE_TEST_PATH)
    df = make_prediction(template_test.filename, model, save=False)
    assert df.shape == PREDICTION_SHAPE
    
    
def test_number_layers_mobilenet():
    """
    Test the number of layers in the MobileNet model.
    """

    model = create_model()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summ_string = "\n".join(stringlist)

    table = stringlist[1:-4][1::2]  # take every other element and remove appendix

    new_table = []
    for entry in table:
        entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
        new_table.append(entry)

    df = pd.DataFrame(new_table[1:], columns=new_table[0])
    number_layers = len(df.index) - 1
    awaited_result = int(NUMBER_OF_LAYERS)

    assert number_layers == awaited_result
