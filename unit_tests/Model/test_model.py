"""Test Model Functionalities"""

import os
import pandas as pd

def test_xySameLength():
    """Ensure image count and train labels are the same length.
    """
    print(os.getcwd())
    trainLabelPath = '../Inputs/Model/labels_train.csv'
    imageTrainPath = '../Inputs/Model/train'

    trainLabels = pd.read_csv(trainLabelPath)
    imageLen = len([file for file in os.listdir(imageTrainPath)])
    assert len(trainLabels.filename) == imageLen
