import tensorflow as tf
from tensorflow.keras import layers, models
from preprocessing import get_data_loaders, BATCH_SIZE, IMG_SIZE

# ── Build the model ────────────────────────────────────────────────────────────

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a Custom CNN from scratch with no pretrained weights.

    This is our baseline model — it learns entirely from our 4,480 training
    images. We use it to establish a performance floor before comparing against
    EfficientNetB0 and ResNet50 which come pretrained on millions of images.

    Args:
        input_shape: height x width x channels — matches our preprocessing output
        num_classes: 4 (glioma, meningioma, notumor, pituitary)
    """

    model = models.Sequential([

        # ── Input ───────────────────────────────────────────────────────────────
        # Tells the model to expect images of shape (224, 224, 3)
        layers.Input(shape=input_shape),

        # ── Block 1 ─────────────────────────────────────────────────────────────
        # Conv2D: applies 32 filters (3x3) that slide across the image.
        # Each filter learns to detect a different low-level feature (edges, curves).
        # padding='same' keeps the output the same size as the input.
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),

        # BatchNormalization: normalizes the output of the conv layer so values
        # stay in a stable range. Speeds up training and helps the model converge.
        layers.BatchNormalization(),

        # MaxPooling2D: shrinks the image by half (2x2 window, take the max value).
        # Keeps the strongest features while reducing computation and overfitting.
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Dropout: randomly turns off 25% of neurons during each training step.
        # Forces the model to not rely on any single neuron — reduces overfitting.
        layers.Dropout(0.25),

        # ── Block 2 ─────────────────────────────────────────────────────────────
        # 64 filters now — the model looks for more complex patterns built on
        # the features detected in block 1 (e.g. combinations of edges = shapes).
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ─────────────────────────────────────────────────────────────
        # 128 filters — even higher-level features like tumor boundaries or textures.
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # ── Block 4 ─────────────────────────────────────────────────────────────
        # 256 filters — the deepest feature extraction layer.
        # By this point the image has been shrunk significantly by the pooling layers,
        # but each remaining unit represents a rich, high-level feature.
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # ── Classifier head ─────────────────────────────────────────────────────
        # Flatten: converts the 3D feature maps (height x width x filters) into
        # a single 1D vector so we can feed it into fully connected layers.
        layers.Flatten(),

        # Dense: a fully connected layer where every neuron connects to every input.
        # 512 neurons learn to combine all the extracted features into a final judgment.
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),

        # Dropout: heavier dropout (50%) before the final output layer to prevent
        # the classifier head from overfitting even if the conv layers generalized well.
        layers.Dropout(0.5),

        # Output layer: 4 neurons, one per class.
        # Softmax converts raw scores into probabilities that sum to 1.
        # e.g. [0.70, 0.10, 0.15, 0.05] → model is 70% confident this is glioma.
        layers.Dense(num_classes, activation='softmax'),
    ])

    return model


# ── Compile the model ──────────────────────────────────────────────────────────

def compile_model(model):
    """
    Configures the model for training by setting the optimizer, loss, and metrics.
    """

    model.compile(
        # Adam: an adaptive learning rate optimizer — one of the most widely used.
        # It adjusts how big each weight update is based on recent gradient history,
        # which makes training faster and more stable than plain gradient descent.
        # lr=0.001 is the standard starting learning rate for Adam.
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

        # Categorical crossentropy: the standard loss function for multiclass classification.
        # It measures how far the model's predicted probabilities are from the true label.
        # A perfect prediction (probability 1.0 on the correct class) gives loss = 0.
        loss='categorical_crossentropy',

        # Track accuracy during training so we can see improvement each epoch.
        metrics=['accuracy'],
    )

    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks():
    """
    Callbacks are functions that run automatically during training at set points
    (e.g. end of each epoch). We use two:
    """

    # EarlyStopping: stops training if validation loss stops improving.
    # Prevents wasting time training extra epochs that don't help — and prevents
    # the model from overfitting by training too long.
    # patience=5 means stop if there's no improvement for 5 consecutive epochs.
    # restore_best_weights=True rolls back to the best epoch's weights at the end.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    )

    # ReduceLROnPlateau: halves the learning rate if val_loss plateaus for 3 epochs.
    # When the model stops improving, a smaller learning rate lets it make finer
    # adjustments rather than overshooting the optimal weights.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # multiply learning rate by 0.5
        patience=3,       # wait 3 epochs before reducing
        min_lr=1e-6,      # never go below this learning rate
    )

    return [early_stopping, reduce_lr]


# ── Train ──────────────────────────────────────────────────────────────────────

def train():
    """
    Loads data, builds the model, and runs training.
    Call this to train the Custom CNN end-to-end.
    """

    # Load the preprocessed data generators from preprocessing.py.
    # All three models use the same function so comparisons are fair.
    train_gen, val_gen, test_gen, class_names = get_data_loaders()

    print(f"Classes: {class_names}")
    print(f"Training on {train_gen.samples} images, validating on {val_gen.samples} images\n")

    # Build and compile the model
    model = build_custom_cnn()
    model = compile_model(model)

    # Print a summary of every layer, its output shape, and parameter count.
    # Useful for understanding the architecture and catching shape mismatches.
    model.summary()

    # Run training.
    # epochs=30 is the maximum — EarlyStopping will likely stop it sooner.
    # Each epoch = one full pass through all 140 training batches.
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )

    # Save the trained model weights to disk so we don't have to retrain.
    model.save("custom_cnn_model.keras")
    print("\nModel saved to custom_cnn_model.keras")

    return model, history, test_gen, class_names


if __name__ == "__main__":
    model, history, test_gen, class_names = train()
