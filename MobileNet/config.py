repo = '/home/jovyan/'
label_train_path = repo + 'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/labels_train.csv'

dataset_path = repo + 'hfactory_magic_folders/colas_data_challenge/computer_vision_challenge/dataset/train/'

IMG_SIZE = 224

CHANNELS = 3

BATCH_SIZE = 256

LR = 1e-5

N_LABELS = 5

EPOCHS = 1

SHUFFLE_BUFFER_SIZE = 1024

THRESH = 0.5

saved_model_path_to_folder = repo + 'hfactory_magic_folders/colas_data_challenge/group_shared_workspace/Models/saved_models/'
folder_name = '28th_model_augustin_LR_0_00001_MobileNetV3_50E_augmented_0_3thresh/'
model_name = 'epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    # "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
    # "https://tfhub.dev/google/bit/m-r101x3/1"
    # "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2"
    # "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5"
    # "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
