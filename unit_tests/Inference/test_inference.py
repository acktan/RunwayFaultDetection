"""Unit tests for the inference function"""

# import pandas as pd
import json
from src.Inference.inference import Inference
from src.Modelling.model import Model

path_conf = "./unit_tests/Params/config_test.json"
conf = json.load(open(path_conf, "r"))


def test_make_prediction():
    """
    Test that the predictions are of the right shape.
    """
    model_class = Model(conf)
    model = model_class.load_my_model()
    infer = Inference(conf, model)
    # template_test = pd.read_csv(conf["inference"]["template_test_path"])
    df = infer.make_prediction()
    shape_0 = int(conf["inference"]["prediction_shape"].split(",")[0])
    shape_1 = int(conf["inference"]["prediction_shape"].split(",")[1])

    assert df.shape == (shape_0, shape_1)
