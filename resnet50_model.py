import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from preprocessing import get_data_loaders, BATCH_SIZE


# ── Build the model ────────────────────────────────────────────────────────────

def build_resnet50(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds a ResNet50 model using transfer learning.

    Same two-phase strategy as EfficientNetB0:
      Phase 1 — freeze the pretrained base, train only the new classifier head
      Phase 2 — unfreeze everything and fine tune at a very low learning rate

    ResNet50 uses skip connections (residual connections) to allow gradients to
    flow directly through the network, which lets it train deeper layers without
    the vanishing gradient problem that plagued earlier deep networks.
    """

    # Load ResNet50 pretrained on ImageNet.
    # include_top=False removes the original 1000-class output layer.
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
    )

    # Freeze the base — lock pretrained weights during Phase 1.
    base_model.trainable = False

    # ── Build the full model ───────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)

    # preprocessing.py rescales pixels to [0, 1], but ResNet50's preprocess_input
    # expects [0, 255] and applies ImageNet mean subtraction (converts to BGR,
    # subtracts channel means [103.939, 116.779, 123.68]).
    # We undo the rescaling and apply ResNet50's preprocessing so the base model
    # receives the same input distribution it was trained on.
    x = layers.Lambda(
        lambda img: tf.keras.applications.resnet.preprocess_input(img * 255.0),
        name='resnet_preprocessing',
    )(inputs)

    # training=False keeps BatchNormalization in inference mode while base is frozen.
    x = base_model(x, training=False)

    # GlobalAveragePooling2D collapses spatial dimensions into a 1D feature vector.
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout to reduce overfitting before the classifier head.
    x = layers.Dropout(0.3)(x)

    # Dense layer to combine ResNet50's features for our 4-class problem.
    x = layers.Dense(256, activation='relu')(x)

    x = layers.Dropout(0.2)(x)

    # Output: 4 probabilities summing to 1 via softmax.
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model, base_model


# ── Phase 1: compile for initial training (base frozen) ───────────────────────

def compile_phase1(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Phase 2: unfreeze and compile for fine tuning ─────────────────────────────

def compile_phase2(model, base_model):
    """
    Unfreeze the base and use a very low learning rate.

    ResNet50 has ~25M parameters — a high learning rate here would overwrite
    the pretrained weights rather than nudging them toward MRI images.
    """
    base_model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks():
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

    model, base_model = build_resnet50()

    # ── Phase 1: train only the classifier head ────────────────────────────────
    print("=" * 60)
    print("Phase 1: Training top layers only (base frozen)")
    print("=" * 60)

    model = compile_phase1(model)
    model.summary()

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

    history_phase2 = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=get_callbacks(),
    )

    model.save("resnet50_model.keras")
    print("\nModel saved to resnet50_model.keras")

    return model, history_phase1, history_phase2, test_gen, class_names


if __name__ == "__main__":
    model, history_phase1, history_phase2, test_gen, class_names = train()
