import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

TRAIN_DIR = 'data/Training'
CLASSES   = ['glioma', 'meningioma', 'notumor', 'pituitary']
LABELS    = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
N_SAMPLES = 5

random.seed(42)

fig, axes = plt.subplots(4, N_SAMPLES, figsize=(14, 11))
fig.suptitle('Sample MRI Images — Brain Tumor Dataset', fontsize=15, fontweight='bold')

for row, (cls, label) in enumerate(zip(CLASSES, LABELS)):
    folder = os.path.join(TRAIN_DIR, cls)
    files  = random.sample(os.listdir(folder), N_SAMPLES)

    for col, fname in enumerate(files):
        img = mpimg.imread(os.path.join(folder, fname))
        axes[row, col].imshow(img, cmap='gray' if img.ndim == 2 else None)
        axes[row, col].axis('off')

    # Add row label as figure text anchored to the left of each row
    ax0 = axes[row, 0]
    fig.text(0.01, ax0.get_position().y0 + ax0.get_position().height / 2,
             label, va='center', ha='left', fontsize=12, fontweight='bold')

plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.05)
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
print('Saved: sample_images.png')
