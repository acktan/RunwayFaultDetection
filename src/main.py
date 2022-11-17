"""Main script to run src"""

import json
import sys
from time import time
sys.path.insert(0, "Evaluation_Model/")
sys.path.insert(0, "Extraction_Airports/")
sys.path.insert(0, "Extraction_Runways/")
sys.path.insert(0, "Inference/")
sys.path.insert(0, "Modelling/")
sys.path.insert(0, "Utils/")

import utils
import inference
import dataloader
import model
import train
import extraction_runways
import evaluation_model
import extraction_airports

def main(logger, conf, img_shape):
    """
    Main function launching step by step the pipeline.
    Args:
        logger: logger file.
        conf: config file.
        img_shape: shape of orthophoto images.
    """
    START = time()
    #extract_airports_class = extraction_airports.Extractionairports(conf, img_shape)
    #extract_airports_class.extract_all_airports()
    #logger.info("Extraction and Saving of Airports Completed...")
    #extract_runways_class = extraction_runways.Extractrunways(conf)
    #extract_runways_class.detect_save_runway()
    #logger.info("Extraction and Saving of Runways Completed...")
    time_1 = time()
    logger.debug("Time for extraction execution :" + str(time_1 - START))
    if conf["model"]["train"] == True:
        logger.info("Training Model...")
        dataset_class = dataloader.DataLoader(conf)
        train_ds, test_ds = dataset_class.create_train_test()
        model_class = model.Model(conf)
        model_mobilenet = model_class.create_model()
        train_class = train.Train(conf, model_mobilenet, train_ds, test_ds)
        history = train_class.training()
        eval_class = evaluation_model.Evaluate(conf, history)
        loss, val_loss, macro_f1, val_macro_f1 = eval_class.learning_curves()
        logger.info(f"The train and validation loss: {loss}, {val_loss}.")
        logger.info(f"The train and validation Macro F1: {macro_f1}, {val_macro_f1}.")
    else:
        logger.info("Loading Model...")
        model_class = model.Model(conf)
        model_mobilenet = model_class.load_my_model()
    # Enter name of submission if save_preds = True in conf.json
    inference_class = inference.Inference(conf, model_mobilenet, "test_subm")
    df = inference_class.make_prediction()
    time_2 = time()
    logger.debug("Time for training/loading model and predicting :" + str(time_2 - time_1))

    
if __name__ == '__main__':
    path_conf = '../Params/Config/config.json'
    conf = json.load(open(path_conf, 'r'))
    img_shape = (25000, 25000)
    path_log = conf['path_log'] # "../log/my_log_file.txt"
    log_level = conf['log_level'] # "DEBUG"
    # instanciation of the logger
    logger = utils.my_get_logger(path_log, log_level, my_name="main_logger")
    try:
        main(logger=logger, conf=conf, img_shape=img_shape)

    except Exception as e:
        logger.error("Error during execution", exc_info=True)