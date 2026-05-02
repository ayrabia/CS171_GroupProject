import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from preprocessing import get_data_loaders, BATCH_SIZE

# ── Build the model ────────────────────────────────────────────────────────────

def build_efficientnet(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds an EfficientNetB0 model using transfer learning.

    We use a two-phase approach:
      Phase 1 — freeze the pretrained base, train only our new output layer
      Phase 2 — unfreeze everything and fine tune at a very low learning rate

    This prevents the random weights in our new output layer from corrupting
    the pretrained weights before our classifier has a chance to stabilize.
    """

    # Load EfficientNetB0 pretrained on ImageNet.
    # include_top=False removes the original 1000-class output layer —
    # we'll add our own 4-class output layer on top instead.
    # weights='imagenet' loads the pretrained weights from Google's training.
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    # Freeze the base model — lock all pretrained weights so they don't
    # change during Phase 1. We only want to train our new layers first.
    base_model.trainable = False

    # ── Build the full model ───────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)

    # preprocessing.py rescales pixels to [0,1], but EfficientNetB0 has its own
    # normalization built into the model and expects raw [0,255] pixel values.
    # We undo the rescaling here so EfficientNet's internal layers see the
    # correct range — without this the base extracts garbage features and the
    # model is stuck at random-chance accuracy (~25%) no matter how long it trains.
    x = layers.Rescaling(scale=255.0)(inputs)

    # training=False keeps BatchNormalization layers in inference mode —
    # important when the base is frozen so running stats don't get corrupted.
    x = base_model(x, training=False)

    # GlobalAveragePooling2D: collapses the spatial dimensions (height x width)
    # by taking the average of each feature map.
    # Converts the 3D output of EfficientNet into a 1D feature vector.
    # More efficient than Flatten and less prone to overfitting.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout before the classifier head to reduce overfitting.
    x = layers.Dropout(0.3)(x)

    # Dense layer to learn how to combine EfficientNet's features for our task.
    x = layers.Dense(128, activation='relu')(x)

    # Dropout again before final output.
    x = layers.Dropout(0.2)(x)

    # Output layer: 4 neurons with softmax — one probability per class.
    # e.g. [0.05, 0.02, 0.03, 0.90] → model is 90% confident this is pituitary.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model, base_model


# ── Phase 1: compile for initial training (base frozen) ───────────────────────

def compile_phase1(model):
    """
    Phase 1 compilation — higher learning rate since we're only training
    the new top layers and the pretrained base is frozen.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Phase 2: unfreeze and compile for fine tuning ─────────────────────────────

def compile_phase2(model, base_model):
    """
    Phase 2 compilation — unfreeze the base and use a very low learning rate.

    The low learning rate (1e-5) is critical here. The pretrained weights are
    already good — we just want to nudge them slightly toward MRI images.
    A high learning rate would destroy the pretrained knowledge.
    """

    # Unfreeze the entire base model so all layers can be updated.
    base_model.trainable = True

    model.compile(
        # 10x smaller learning rate than Phase 1 — fine tuning requires
        # much smaller steps to avoid overwriting the pretrained weights.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks():
    """
    Same callback strategy as the Custom CNN:
    - EarlyStopping: stop if val_loss doesn't improve for 5 epochs
    - ReduceLROnPlateau: halve the learning rate if val_loss plateaus for 3 epochs
    """

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    return [early_stopping, reduce_lr]


# ── Train ──────────────────────────────────────────────────────────────────────

def train():
    """
    Full two-phase training for EfficientNetB0.
    """

    train_gen, val_gen, test_gen, class_names = get_data_loaders()

    print(f"Classes: {class_names}")
    print(f"Training on {train_gen.samples} images, validating on {val_gen.samples} images\n")

    # Build the model
    model, base_model = build_efficientnet()

    # ── Phase 1: train only the top layers ────────────────────────────────────
    print("=" * 60)
    print("Phase 1: Training top layers only (base frozen)")
    print("=" * 60)

    model = compile_phase1(model)
    model.summary()

    # Run Phase 1 for up to 10 epochs.
    # Since only a few layers are trainable this is fast and just gets
    # our output layer to a reasonable starting point.
    history_phase1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )

    # ── Phase 2: fine tune the whole network ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Fine tuning entire network (base unfrozen)")
    print("=" * 60)

    model = compile_phase2(model, base_model)

    # Run Phase 2 for up to 20 more epochs.
    # EarlyStopping will cut this short if val_loss stops improving.
    history_phase2 = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )

    # Save the trained model
    model.save("efficientnet_model.keras")
    print("\nModel saved to efficientnet_model.keras")

    return model, history_phase1, history_phase2, test_gen, class_names


if __name__ == "__main__":
    model, history_phase1, history_phase2, test_gen, class_names = train()
