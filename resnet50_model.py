import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from preprocessing import get_data_loaders, BATCH_SIZE

# ── Build the model ────────────────────────────────────────────────────────────

def build_resnet50(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a ResNet50 model using transfer learning.

    ResNet50 uses residual connections (skip connections) that allow gradients
    to flow directly through the network, enabling much deeper architectures
    without vanishing gradient problems. It has 50 layers vs EfficientNetB0's
    more compact design — we're testing whether the extra depth helps with MRI.

    We use the same two-phase approach as EfficientNetB0:
      Phase 1 — freeze the pretrained base, train only our new output layer
      Phase 2 — unfreeze everything and fine tune at a very low learning rate
    """

    # Load ResNet50 pretrained on ImageNet.
    # include_top=False removes the original 1000-class output layer —
    # we'll add our own 4-class output layer on top instead.
    # weights='imagenet' loads the pretrained weights from Microsoft Research.
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    # Freeze the base model — lock all pretrained weights so they don't
    # change during Phase 1. We only want to train our new layers first.
    base_model.trainable = False

    # ── Build the full model ───────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)

    # preprocessing.py rescales to [0,1] but ResNet50 expects [0,255] with
    # ImageNet mean subtraction. We undo the rescaling then subtract the
    # per-channel ImageNet means using a Normalization layer (fully serializable).
    # Using preprocess_input() as a plain function call causes a Keras
    # serialization error when saving to .keras format.
    x = layers.Rescaling(scale=255.0)(inputs)
    x = layers.Normalization(
        mean=[123.68, 116.779, 103.939],
        variance=[1.0, 1.0, 1.0],
    )(x)

    # training=False keeps BatchNormalization layers in inference mode —
    # important when the base is frozen so running stats don't get corrupted.
    x = base_model(x, training=False)

    # GlobalAveragePooling2D: collapses the spatial dimensions (height x width)
    # by taking the average of each feature map.
    # Converts the 3D output of ResNet50 into a 1D feature vector.
    # More efficient than Flatten and less prone to overfitting.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout before the classifier head to reduce overfitting.
    x = layers.Dropout(0.3)(x)

    # Dense layer to learn how to combine ResNet50's features for our task.
    # 256 units since ResNet50 is deeper and produces richer features than
    # EfficientNetB0 (which used 128).
    x = layers.Dense(256, activation='relu')(x)

    # BatchNormalization to stabilize training of the dense layer.
    x = layers.BatchNormalization()(x)

    # Dropout again before final output.
    x = layers.Dropout(0.2)(x)

    # Output layer: 4 neurons with softmax — one probability per class.
    # e.g. [0.10, 0.75, 0.05, 0.10] → model is 75% confident this is meningioma.
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
    Same callback strategy as the other models:
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
    Full two-phase training for ResNet50.
    """

    train_gen, val_gen, test_gen, class_names = get_data_loaders()

    print(f"Classes: {class_names}")
    print(f"Training on {train_gen.samples} images, validating on {val_gen.samples} images\n")

    # Build the model
    model, base_model = build_resnet50()

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
    model.save("resnet50_model.keras")
    print("\nModel saved to resnet50_model.keras")

    return model, history_phase1, history_phase2, test_gen, class_names


if __name__ == "__main__":
    model, history_phase1, history_phase2, test_gen, class_names = train()
