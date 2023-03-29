# Configurations for Mask R-CNN
import os

class PennFudanConfig(Config):
    """Configuration for training on the PennFudan dataset.
    Derives from the base Config class and overrides values specific
    to the PennFudan dataset.
    """
    # Give the configuration a recognizable name
    NAME = "pennfudan"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + pedestrian

    # All of our training images are 640x640
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 200

    # This is how often validation is run. If you are using too much hard drive
    # space you can set this to a higher number to save space.
    VALIDATION_STEPS = 5

    # Matterport originally used resnet101, but we're using resnet50 here
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 10
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
