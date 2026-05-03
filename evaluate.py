import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import get_data_loaders

# Models to evaluate — name shown in output, path to saved .keras file
MODELS = [
    ("Custom CNN",    "custom_cnn_model.keras"),
    ("EfficientNetB0","efficientnet_model.keras"),
    ("ResNet50",      "resnet50_model.keras"),
]


def get_predictions(model, test_gen):
    """
    Run the model over the entire test set and return true vs predicted labels.

    test_gen must have shuffle=False (it does — see preprocessing.py) so the
    order of predictions lines up with test_gen.classes.
    """
    # Reset to the start of the generator — important when evaluating multiple
    # models in sequence, since the generator keeps a position internally.
    test_gen.reset()

    # model.predict returns a probability array of shape (n_samples, 4).
    # np.argmax picks the class with the highest probability for each image.
    probs  = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_gen.classes   # ground-truth integer labels in the same order

    return y_true, y_pred


def print_metrics(y_true, y_pred, class_names, model_name):
    """
    Print accuracy, per-class precision/recall/F1, and macro F1 for one model.
    Macro F1 is our primary metric — it treats every class equally regardless
    of how many samples it has, which matters because meningioma is slightly
    underrepresented and misclassifying a tumor as 'no tumor' is costly.
    """
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    # classification_report prints per-class precision, recall, F1, and support,
    # plus macro and weighted averages at the bottom.
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Also surface the macro F1 on its own line so it's easy to find.
    from sklearn.metrics import f1_score, accuracy_score
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1      : {macro_f1:.4f}")


def plot_confusion_matrices(cms, class_names, model_names):
    """
    Plot one confusion matrix per model side by side and save to a PNG.
    Each cell shows the raw count of images predicted as that class.
    """
    fig, axes = plt.subplots(1, len(cms), figsize=(6 * len(cms), 5))

    for ax, cm, name in zip(axes, cms, model_names):
        # Normalize the color scale per row so each true class fills the full
        # color range — makes it easy to spot where each class gets confused.
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

        # Annotate each cell with the raw count.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_norm[i, j] > 0.6 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=11, color=color)

        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha='right')
        ax.set_yticklabels(class_names)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
    print("\nConfusion matrices saved to confusion_matrices.png")
    plt.show()


def main():
    # Load the shared test generator — same preprocessing as training so the
    # comparison between models is fair.
    _, _, test_gen, class_names = get_data_loaders()

    print(f"Evaluating on {test_gen.samples} test images")
    print(f"Classes: {class_names}")

    cms         = []
    model_names = []

    for model_name, model_path in MODELS:
        print(f"\nLoading {model_name} from {model_path} ...")
        # compile=False skips restoring the optimizer — we only need the
        # weights for inference, not the training state.
        model = tf.keras.models.load_model(model_path, compile=False)

        y_true, y_pred = get_predictions(model, test_gen)
        print_metrics(y_true, y_pred, class_names, model_name)

        cms.append(confusion_matrix(y_true, y_pred))
        model_names.append(model_name)

    plot_confusion_matrices(cms, class_names, model_names)


if __name__ == "__main__":
    main()
