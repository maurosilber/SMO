from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from scipy.ndimage._ni_support import _normalize_sequence


def _euclidean_norm(x: list[np.ndarray]) -> np.ndarray:
    return np.sqrt(sum(xi ** 2 for xi in x))


def _normalized_gradient(input):
    """Calculates the normalized gradient of a scalar field.

    Parameters
    ----------
    input : numpy.array
        Input field.

    Returns
    -------
    numpy.array
        The normalized gradient of the scalar field.
    """
    grad = np.gradient(input.astype(float, copy=False))

    if input.ndim == 1:
        return np.sign(grad, out=grad)

    norm = _euclidean_norm(grad)
    mask = norm > 0
    for x in grad:
        np.divide(x, norm, where=mask, out=x)
    return grad


def smo(input: np.ndarray, *, sigma: float, size: int):
    """Applies the Silver Mountain Operator (SMO) to a scalar field.

    Parameters
    ----------
    input : numpy.array
        Input field.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.

    Returns
    -------
    numpy.array
    """
    size = _normalize_sequence(size, input.ndim)
    input = ndimage.gaussian_filter(input.astype(float, copy=False), sigma=sigma)
    grad = _normalized_gradient(input)
    # Inplace average gradient
    for x in grad:
        ndimage.uniform_filter(x, size=size, output=x)
    return _euclidean_norm(grad)


def smo_rv(shape, *, sigma, size, random_state=None):
    """Generates a random variable of the SMO operator for a given sigma and size.

    Parameters
    ----------
    shape : tuple of ints
        Dimension of the random image to be generated.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.
    random_state : numpy.random.Generator
        By default

    Returns
    -------
    HistogramRV
        Subclass of scipy.stats.rv_histogram.
    """
    if random_state is None:
        random_state = np.random.default_rng(seed=42)

    image = random_state.uniform(size=shape)
    smo_image = smo(image, sigma=sigma, size=size)
    hist = np.histogram(smo_image, bins="fd")
    return stats.rv_histogram(hist)
