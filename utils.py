import cv2
import numpy as np
from skimage.measure import label

# --------------------------------------------------
# Preprocessing (Grayscale MRI)
# --------------------------------------------------
def preprocess_image(image, target_size=(128, 128)):
    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)  # (128,128,1)
    image = np.expand_dims(image, axis=0)   # (1,128,128,1)

    return image


# --------------------------------------------------
# Post-processing (Largest Component Only)
# --------------------------------------------------
def keep_largest_component(mask):
    labeled_mask = label(mask)

    if labeled_mask.max() == 0:
        return mask

    largest_component = max(
        range(1, labeled_mask.max() + 1),
        key=lambda x: np.sum(labeled_mask == x)
    )

    return (labeled_mask == largest_component).astype(np.uint8)


