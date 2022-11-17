"""Load and Prepare Data for Model"""

import logging
import warnings
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logger = logging.getLogger("main_logger")


class DataLoader:
    """Create a class to extract "runways from cropped airports."""

    def __init__(self, conf):
        self.conf = conf
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

    def read_split_data(self):
        """Read the training data and split in train and validation.

        Args:
            conf: the config file
        Returns:
            X_train, y_train: The training data and label
            X_test, y_test: The validation data and label
        """
        logger.info("Reading the data...")
        conf_path = self.conf["paths"]
        path_train = conf_path["model_input_path"] + \
            conf_path["train_label_file"]
        labels_train = pd.read_csv(path_train)
        logger.info("Splitting the data in train and validation...")
        X_train, X_test, y_train, y_test = train_test_split(
            labels_train["filename"],
            labels_train.drop("filename", axis=1),
            test_size=0.2,
            random_state=44,
        )
        DATASET_PATH = conf_path["model_input_path"] + \
            conf_path["folder_train"]
        X_train = [os.path.join(DATASET_PATH, str(f)) for f in X_train]
        X_test = [os.path.join(DATASET_PATH, str(f)) for f in X_test]

        y_train = y_train.values
        y_test = y_test.values
        return X_train, y_train, X_test, y_test

    def parse_function(self, filename, label, max_delta=0.2, seed=(1, 2)):
        """Function that returns a tuple of normalized image array and labels array.
        Args:
            filename: string representing path to image
            label: 0/1 one-dimensional array of size N_LABELS,
            max_delta (float),
            seed (tensor): a shape [2] Tensor, the seed to the
                random number generator. Must have dtype int32 or int64.

        Returns:
            image_augmented
                (tensorflow.python.data.ops.dataset_ops.ParallelMapDataset)
            label (numpy.ndarray): array of labels
        """
        channels = self.conf["model"]["CHANNELS"]
        img_size = self.conf["model"]["IMG_SIZE"]
        # Read an image from a file
        image_string = tf.io.read_file(filename)
        # Decode it into a dense vector
        image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
        # Resize it to fixed shape
        image_resized = tf.image.resize(image_decoded, [img_size, img_size])
        # Normalize it from [0, 255] to [0.0, 1.0]
        image_normalized = image_resized / 255.0
        # Random flip
        image_augmented = tf.image.stateless_random_flip_left_right(
            image_normalized, seed=seed
        )
        image_augmented = tf.image.stateless_random_flip_up_down(
            image_augmented, seed
        )

        return image_augmented, label

    def create_dataset(self, filenames, labels, batch_size, is_training=True):
        """Load and parse dataset.
        Args:
            filenames: list of image paths
            labels: numpy array of shape (batch_size, N_LABELS)
            batch_size (int): number of samples that will be
                propagated through the network,
            is_training: boolean to indicate training mode

        Returns:
            dataset (tensorflow.python.data.ops.dataset_ops.PrefetchDataset)
        """
        logger.info("Creating the datasets...")
        # Create a first dataset of file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # Parse and preprocess observations in parallel
        dataset = dataset.map(
            self.parse_function, num_parallel_calls=self.AUTOTUNE
        )

        if is_training:
            # This is a small dataset, only load it once,
            # and keep it in memory.
            dataset = dataset.cache()
            # Shuffle the data each buffer size
            dataset = dataset.shuffle(buffer_size=len(filenames))

        # Batch the data for multiple steps
        dataset = dataset.batch(batch_size)
        # Fetch batches in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)

        return dataset

    def create_train_test(self):
        """Create the train and validation datasets.

        Args:
            conf: config file model args.
        Returns:
            train_ds: preprocessed train dataset.
            test_ds: preprocessed test dataset.
        """
        X_train, y_train, X_test, y_test = self.read_split_data()
        batch_size = self.conf["model"]["BATCH_SIZE"]
        train_ds = self.create_dataset(
            filenames=X_train, labels=y_train,
            batch_size=batch_size, is_training=True
        )
        test_ds = self.create_dataset(
            filenames=X_test, labels=y_test,
            batch_size=batch_size, is_training=True
        )
        return train_ds, test_ds
