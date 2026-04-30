import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  Constants 

# All three models expect 224x224 input.
# EfficientNetB0 and ResNet50 are pretrained on ImageNet at this size,
# so we standardize the custom CNN to the same size for a fair comparison.
IMG_SIZE = (224, 224)

# How many images to process at a time during training.
# 32 is a standard default — small enough to fit in memory, large enough
# for stable gradient estimates.
BATCH_SIZE = 32

# 20% of the training set is held out as a validation set.
# The test set is kept completely untouched until final evaluation.
VALIDATION_SPLIT = 0.2

# Paths to the dataset folders (produced by download_dataset.py)
TRAIN_DIR = os.path.join("data", "Training")
TEST_DIR  = os.path.join("data", "Testing")

# ─ Augmentation for the training set

def get_train_generator(train_dir=TRAIN_DIR, batch_size=BATCH_SIZE):
    """
    Loads training images with augmentation and splits off a validation set.

    Augmentation means we randomly transform each image every epoch so the
    model sees slight variations — flipped, rotated, zoomed, etc.
    This prevents overfitting: the model learns the general shape of a tumor
    rather than memorising specific pixel patterns in the training images.

    Augmentation is applied ONLY to training data, not validation or test,
    so that evaluation scores reflect real-world performance.
    """

    train_datagen = ImageDataGenerator(
        # Scale pixel values from [0, 255] to [0, 1].
        # Large raw pixel values make training numerically unstable;
        # normalising keeps activations in a well-behaved range.
        rescale=1.0 / 255,

        # Randomly flip images left-right.
        # Brain MRIs can be oriented either way, so this is medically valid.
        horizontal_flip=True,

        # Randomly rotate up to 10 degrees.
        # Small rotations mimic natural variation in how scans are positioned.
        rotation_range=10,

        # Randomly zoom in or out by up to 10%.
        zoom_range=0.1,

        # Randomly shift brightness by up to 10%.
        # Helps the model handle variation in scanner contrast settings.
        brightness_range=[0.9, 1.1],

        # Reserve 20% of training images for validation.
        # This subset is NOT augmented (see below) — it stays clean so we
        # can honestly measure how well the model is learning each epoch.
        validation_split=VALIDATION_SPLIT,
    )

    # flow_from_directory reads images from subfolders and assigns class labels
    # automatically based on folder names: glioma, meningioma, notumor, pituitary.
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,       # resize every image to 224x224
        batch_size=batch_size,
        class_mode="categorical",   # one-hot encode labels for 4-class softmax output
        subset="training",          # use the 80% training portion
        shuffle=True,               # shuffle order each epoch to reduce ordering bias
        seed=42,                    # fixed seed so results are reproducible
    )

    return train_generator


def get_val_generator(train_dir=TRAIN_DIR, batch_size=BATCH_SIZE):
    """
    Loads the validation split of the training set WITHOUT augmentation.

    Validation images must not be augmented — we want to measure the model's
    performance on clean, unmodified images so the metric is meaningful.
    """

    # Same rescaling as training, but no random transforms.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
    )

    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",        # use the 20% held-out portion
        shuffle=False,              # no shuffling for validation — order doesn't matter
        seed=42,
    )

    return val_generator


def get_test_generator(test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Loads the test set WITHOUT augmentation or shuffling.

    This set is only used once — after all training is done — to produce the
    final accuracy, F1, and confusion matrix numbers we report in the paper.
    Never use test data during training or hyperparameter tuning.
    """

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,              # keep order fixed so confusion matrix rows line up
    )

    return test_generator


def get_data_loaders(train_dir=TRAIN_DIR, test_dir=TEST_DIR, batch_size=BATCH_SIZE):
    """
    Convenience function that returns all three data generators at once.

    All three models call this single function so that preprocessing is
    identical across the custom CNN, EfficientNetB0, and ResNet50 —
    making the comparison between them fair and apples-to-apples.

    Returns:
        train_gen  — augmented training batches
        val_gen    — clean validation batches (same images as train, different split)
        test_gen   — clean test batches (completely separate images)
        class_names — list of class labels in the order the model outputs them
    """
    train_gen = get_train_generator(train_dir, batch_size)
    val_gen   = get_val_generator(train_dir, batch_size)
    test_gen  = get_test_generator(test_dir, batch_size)

    # class_indices maps folder name → integer label, e.g. {"glioma": 0, ...}
    # We invert it to get a list of names ordered by label index.
    class_names = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]

    return train_gen, val_gen, test_gen, class_names
