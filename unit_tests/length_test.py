import os

def xySameLength():
    """
    Count train image count and trainLabels file count
    """
    trainLabelPath = '../../hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/labels_train.csv'
    imageTrainPath = '../../hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/train/'

    trainLabels = pd.read_csv(trainLabelPath)
    imageLen = len([file for file in os.listdir(path)])
    assert len(trainLabels.filename) == imageLen
    
