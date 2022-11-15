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

labels_train = pd.read_csv(label_train_path)

X_train, X_test, y_train, y_test = train_test_split(labels_train['filename'],
                                                    labels_train.drop('filename', axis=1),
                                                    test_size=0.2,
                                                    random_state=44)

X_train = [os.path.join(dataset_path, str(f)) for f in X_train]
X_test = [os.path.join(dataset_path, str(f)) for f in X_test]

y_train = y_train.values
y_test = y_test.values

train_ds = create_dataset(X_train, y_train,
                          BATCH_SIZE = BATCH_SIZE,
                          SHUFFLE_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE)
test_ds = create_dataset(X_test, y_test,
                         BATCH_SIZE = BATCH_SIZE,
                         SHUFFLE_BUFFER_SIZE = SHUFFLE_BUFFER_SIZE)

model = create_model(IMG_SIZE=IMG_SIZE,
                     CHANNELS=CHANNELS,
                     LR=LR,
                     N_LABELS=N_LABELS)

test = (X_test, y_test)

history = training(model,
         train_ds,
         EPOCHS,
         test,
         saved_model_path_to_folder + folder_name,
         model_name
        )

losses, val_losses, macro_f1s, val_macro_f1s = learning_curves(history)
