import os
import pandas as pd

def test_xySameLength():
    """
    Count train image count and trainLabels file count
    
    FIX: update paths later 
    """
    print(os.getcwd())
    trainLabelPath = 'Inputs/Model/labels_train.csv'
    imageTrainPath = 'Inputs/Model/train'

    trainLabels = pd.read_csv(trainLabelPath)
    imageLen = len([file for file in os.listdir(imageTrainPath)])
    assert len(trainLabels.filename) == imageLen
    
