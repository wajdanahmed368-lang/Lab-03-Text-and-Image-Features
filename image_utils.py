"""
Image Feature Extraction Utilities
Course: Introduction to Artificial Intelligence (AI-101L)
Laboratory 3: Transforming Textual and Image Data into Machine-Understandable Formats
"""
import numpy as np
from PIL import Image

def image_to_features(path_or_image, size=(64, 64)):
    """
    Load, resize, convert to grayscale, and flatten an image into a 1-D feature vector.

    Parameters
    ----------
    path_or_image : str or PIL.Image.Image
        Filepath to the image, or an existing PIL Image object.
    size : tuple[int, int], default=(64, 64)
        Target (width, height) for image resizing. Enforces fixed feature length.

    Returns
    -------
    numpy.ndarray
        1-D flattened array of shape (size[0] * size[1],) and dtype uint8.
    """
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image)
    else:
        img = path_or_image

    img = img.convert("RGB")     # Normalize to 3-channel RGB
    img = img.resize(size)       # Enforce consistent dimensions
    img = img.convert("L")        # Grayscale conversion (3:1 feature reduction)
    return np.array(img).flatten()

if __name__ == "__main__":
    dummy = Image.fromarray(np.random.randint(0, 256, (120, 80, 3), dtype=np.uint8), "RGB")
    vec = image_to_features(dummy, size=(64, 64))
    print(f"Extracted feature vector length: {vec.shape[0]} (Expected: {64*64})")
