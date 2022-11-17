import logging
import os
import warnings

import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)


from tools.train import training, learning_curves
from tools.dataset import create_dataset
from tools.model import create_model

from config import *

labels_train = pd.read_csv(LABEL_TRAIN_PATH)

X_train, X_test, y_train, y_test = train_test_split(labels_train['filename'],
                                                    labels_train.drop('filename', axis=1),
                                                    test_size=TEST_SIZE,
                                                    random_state=44
                                                   )

X_train = [os.path.join(DATASET_PATH, str(f)) for f in X_train]
X_test = [os.path.join(DATASET_PATH, str(f)) for f in X_test]

y_train = y_train.values
y_test = y_test.values

train_ds = create_dataset(filenames=X_train, 
                          labels=y_train, 
                          batch_size=BATCH_SIZE, 
                          is_training=True
                         )
test_ds = create_dataset(filenames=X_test, 
                         labels=y_test, 
                         batch_size=BATCH_SIZE, 
                         is_training=True
                        )

model = create_model(feature_extractor_url=FEATURE_EXTRACTOR_URL, 
                     img_size=IMG_SIZE, 
                     channels=CHANNELS, 
                     lr=LR, 
                     n_labels=N_LABELS
                    )

history = training(model=model, 
                   train_ds=train_ds, 
                   test_ds=test_ds, 
                   path_to_saved_models_folder=SAVED_MODELS_PATH_TO_FOLDERS+FOLDER_NAME, 
                   model_name=MODEL_NAME, 
                   epochs=EPOCHS, 
                   batch_size=BATCH_SIZE
        )

losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)
