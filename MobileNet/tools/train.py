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


def training(model, train_ds, EPOCHS, test, saved_model_folder, model_name, BATCH_SIZE, SHUFFLE_BUFFER_SIZE):
    start = time()
    validation_dataset = create_dataset(test[0], test[1],
                                     BATCH_SIZE=BATCH_SIZE,
                                     SHUFFLE_BUFFER_SIZE=SHUFFLE_BUFFER_SIZE)
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=validation_dataset,
                        # callbacks=[
                        #         tf.keras.callbacks.ModelCheckpoint(
                        #             os.path.join(saved_model_folder, model_name),
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
    plt.plot(range(1, epochs+1), macro_f1, label='Training Macro F1-score')
    plt.plot(range(1, epochs+1), val_macro_f1, label='Validation Macro F1-score')
    plt.legend(loc='lower right')
    plt.ylabel('Macro F1-score')
    plt.title('Training and Validation Macro F1-score')
    plt.xlabel('epoch')

    plt.show()

    return loss, val_loss, macro_f1, val_macro_f1
