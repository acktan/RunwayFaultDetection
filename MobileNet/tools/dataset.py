import logging
import warnings

import tensorflow as tf



warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

AUTOTUNE = tf.data.experimental.AUTOTUNE

from config import *

def parse_function(filename, label, channels = CHANNELS, img_size = IMG_SIZE, max_delta = 0.2, seed=(1, 2)):
    """Function that returns a tuple of normalized image array and labels array.
    Args:
        filename: string representing path to image
        label: 0/1 one-dimensional array of size N_LABELS, 
        channels (int): number of channel, 3 by default when rgb,
        img_size (int): image height and width,
        max_delta (float), 
        seed (tensor): a shape [2] Tensor, the seed to the random number generator. Must have dtype int32 or int64.
    
    Returns:
        image_augmented (tensorflow.python.data.ops.dataset_ops.ParallelMapDataset)
        label (numpy.ndarray): array of labels
    """
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [img_size, img_size])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    # Random brightness.
    # image_augmented = tf.image.stateless_random_brightness(
    #     image_normalized, max_delta=max_delta, seed=seed)
    # image_augmented = tf.image.stateless_random_contrast(
    #     image_augmented, lower=0.4, upper=0.6, seed=seed)
    # image_augmented = tf.image.stateless_random_jpeg_quality(
    #     image_augmented, min_jpeg_quality=75, max_jpeg_quality=100, seed=seed)
    image_augmented = tf.image.stateless_random_flip_left_right(
        image_normalized, seed=seed)
    image_augmented = tf.image.stateless_random_flip_up_down(
        image_augmented, seed)

    return image_augmented, label
    # return image_normalized, label


def create_dataset(filenames, labels, batch_size=BATCH_SIZE, is_training=True):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (batch_size, N_LABELS)
        batch_size (int): number of samples that will be propagated through the network,  
        is_training: boolean to indicate training mode
    
    Returns:
        dataset (tensorflow.python.data.ops.dataset_ops.PrefetchDataset) 
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=len(filenames))

    # Batch the data for multiple steps
    dataset = dataset.batch(batch_size)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
