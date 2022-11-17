REPO = '/home/jovyan/'
LABEL_TRAIN_PATH = REPO + 'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/labels_train.csv'

DATASET_PATH = REPO + 'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/train/'

IMG_SIZE = 224

CHANNELS = 3

BATCH_SIZE = 256

LR = 1e-4

N_LABELS = 5

EPOCHS = 2

THRESH = 0.5

PREDICTION_THRESH = 0.4

SAVED_MODELS_PATH_TO_FOLDERS = REPO + 'hfactory_magic_folders/colas_data_challenge/group_shared_workspace/Models/saved_models/'
FOLDER_NAME = '28th_model_augustin_LR_0_00001_MobileNetV3_50E_augmented_0_3thresh/'
MODEL_NAME = 'epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'

TESTING_PATH = REPO + 'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/test'

PATH_TO_SAVED_SUBMISSIONS_FOLDER = REPO + 'hfactory_magic_folders/colas_data_challenge/group_shared_workspace/Models/submissions/'
SUBMISSION_NAME = '43th_final_model_to_submit.csv'

FEATURE_EXTRACTOR_URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    # "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
    # "https://tfhub.dev/google/bit/m-r101x3/1"
    # "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2"
    # "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    # "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
