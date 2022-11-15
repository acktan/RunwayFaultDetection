import os
import pandas as pd

def test_xySameLength():
    """
    Count train image count and trainLabels file count
    
    FIX: update paths later 
    """
    trainLabelPath = '/home/jovyan/hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/labels_train.csv'
    imageTrainPath = '/home/jovyan/hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/train/'

    trainLabels = pd.read_csv(trainLabelPath)
    imageLen = len([file for file in os.listdir(imageTrainPath)])
    assert len(trainLabels.filename) == imageLen
    
