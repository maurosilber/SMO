import numpy as np
from skimage.io import imread


def load_image(file):
    """Loads and crops an image."""
    return _crop(imread(file))


def _crop(image):
    """Crops the 40% center region of the image."""
    n = len(image)
    x0, x1 = int(0.3 * n), int(0.7 * n)
    return image[x0:x1, x0:x1]


def image_histogram(image) -> np.array:
    return np.bincount(image.flat)


def median_from_histogram(histogram) -> int:
    cdf = np.cumsum(histogram)
    return np.searchsorted(cdf, cdf[-1] / 2)
