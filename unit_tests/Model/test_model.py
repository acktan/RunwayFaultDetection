"""Test Model Functionalities"""

import re
import pandas as pd
import json
from src.Modelling.dataloader import DataLoader
from src.Modelling.model import Model

path_conf = "./unit_tests/Params/config_test.json"
conf = json.load(open(path_conf, "r"))


def test_create_dataset():
    """
    Test that the dataset are well created and according to the
        parameters we specified.
    """
    BATCH_SIZE = conf["model"]["BATCH_SIZE"]
    IMG_SIZE = conf["model"]["IMG_SIZE"]
    CHANNELS = conf["model"]["CHANNELS"]
    N_LABELS = conf["model"]["N_LABELS"]
    dataset_class = DataLoader(conf)
    train_ds, _ = dataset_class.create_train_test()

    for f, l in train_ds.take(1):
        assert f.numpy().shape == (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)
        assert l.numpy().shape == (BATCH_SIZE, N_LABELS)


def test_number_layers_mobilenet():
    """
    Test the number of layers in the MobileNet model.
    """
    NUMBER_OF_LAYERS = conf["model"]["NUMBER_OF_LAYERS"]
    model_class = Model(conf)
    model = model_class.create_model()
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))

    # take every other element and remove appendix
    table = stringlist[1:-4][1::2]

    new_table = []
    for entry in table:
        entry = re.split(r"\s{2,}", entry)[:-1]  # remove whitespace
        new_table.append(entry)

    df = pd.DataFrame(new_table[1:], columns=new_table[0])
    number_layers = len(df.index) - 1
    awaited_result = int(NUMBER_OF_LAYERS)

    assert number_layers == awaited_result
