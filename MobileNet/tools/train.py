import logging
import os
import warnings
from time import time

import matplotlib.pyplot as plt
import matplotlib.style as style
import tensorflow as tf

import keras

from config import *
from tools.dataset import create_dataset


warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def training(model, train_ds, test_ds, path_to_saved_models_folder, model_name, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """trains a deep learning model based on Keras tensorflow

    Args:
        model (keras.engine.sequential.Sequential)
        train_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset), 
        test_ds (tensorflow.python.data.ops.dataset_ops.PrefetchDataset), 
        path_to_saved_models_folder (str), 
        model_name (str), 
        epochs (int): complete pass through the whole training set, 
        batch_size (int): number of samples that will be propagated through the network
    
    Returns:
        history: history callback of fitting a tensorflow keras model
    """
    start = time()
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=test_ds,
                        # callbacks=[
                        #         tf.keras.callbacks.ModelCheckpoint(
                        #             os.path.join(path_to_saved_models_folder, model_name),
                        #             monitor='val_loss', save_best_only=True
                        #             )
                        #         ]
                       )
    print(f'Training took {round(time()-start, 2)} sec')
    return history


def learning_curves(history):
    """Plot the learning curves of loss and macro f1 score
    for the training and validation datasets.

    Args:
        history: history callback of fitting a tensorflow keras model
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    macro_f1 = history.history['macro_f1']
    val_macro_f1 = history.history['val_macro_f1']

    epochs = len(loss)

    style.use("bmh")
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs+1), loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs+1), macro_f1, label='Training F1-score')
    plt.plot(range(1, epochs+1), val_macro_f1, label='Validation F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('F1-score')
    plt.title('Training and Validation F1-score')
    plt.xlabel('epoch')

    plt.show()

    return loss, val_loss, macro_f1, val_macro_f1
