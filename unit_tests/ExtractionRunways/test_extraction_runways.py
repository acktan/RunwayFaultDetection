"""Unit tests for extraction runway functions"""

import json
import cv2
import numpy as np
from src.Extraction_Runways.extraction_runways import Extractrunways

# sys.path.insert(0, "../src/Extraction_Runways/")

# import extraction_runways

path_conf = "./unit_tests/Params/config_test.json"
conf = json.load(open(path_conf, "r"))


def test_get_labeled_runways():
    """Test the reading and preparation of runways file.

    Steps:
        Create a class Extraction Runways
        Read and get the dataframe for runways.
    Output:
        lend(df) should be 10.
    """
    extraction_runways_class = Extractrunways(conf)
    df = extraction_runways_class.get_labeled_runways()
    assert len(df) == 10


def test_calculate_coordinates():
    """Test the calculation of coordinates for runways.

    Steps:
        Create a class Extraction Runways
        Read and get the dataframe for runways.
        Get the label for one runway:
        {'x': 23.324089474036477,
        'y': 73.76001569145896,
        'width': 101.71179139083763,
        'height': 3.607959056568356,
        'rotation': 295.08359400619156,
        'rectanglelabels': ['Runway'],
        'original_width': 13141,
        'original_height': 18310}
        Get the coordinates.
    Output:
        cnt: [[[9329,  1680]],
              [[3663, 13785]],
              [[3065, 13505]],
              [[8731, 1400]]]
    """

    extraction_runways_class = Extractrunways(conf)
    df = extraction_runways_class.get_labeled_runways()
    label = df.loc[0, "label"]
    cnt = extraction_runways_class.calculate_coordinates(label)
    assert cnt[0][0][0] == 9329
    assert cnt[0][0][1] == 1680
    assert cnt[1][0][0] == 3663
    assert cnt[1][0][1] == 13785
    assert cnt[2][0][0] == 3065
    assert cnt[2][0][1] == 13505
    assert cnt[3][0][0] == 8731
    assert cnt[3][0][1] == 1400


def test_crop_save_runway():
    """Test the cropping and saving of runways.

    Steps:
        Create a class Extraction Runways
        Read and get the dataframe for runways.
        Get the label for one runway:
        Get the coordinates.
        Get the img_name.
        Crop and Save image
    Output:
        None, image should be saved in Outputs_Test file.
    """
    extraction_runways_class = Extractrunways(conf)
    df = extraction_runways_class.get_labeled_runways()
    label = df.loc[
        np.where(
            df["image"] == "cropped_cv2_93-2021-0670-6860-LA93-0M20-E080_.jpg"
        ),
        "label",
    ].values[0]
    image_name = "cropped_cv2_93-2021-0670-6860-LA93-0M20-E080_.jpg"
    index = 0
    path_output = (
        conf["paths"]["Outputs_path"]
        + conf["paths"]["Outputs_test_path"]
        + conf["paths"]["runways_extraction_file"]
    )
    cnt = extraction_runways_class.calculate_coordinates(label)
    rect = cv2.minAreaRect(cnt)
    output = extraction_runways_class.crop_save_runway(
        rect, image_name, path_output, index
    )
    assert output is None
