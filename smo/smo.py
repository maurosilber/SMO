from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import rv_continuous, rv_histogram


def _rv(x: np.ndarray) -> rv_continuous:
    hist = np.histogram(x.flat, bins="auto")
    return rv_histogram(hist)


def _euclidean_norm(x: list[np.ndarray]) -> np.ndarray:
    if len(x) == 1:
        return np.abs(x[0])

    return np.sqrt(sum(xi ** 2 for xi in x))


def _filter(
    filter: callable, image: np.ndarray | np.ma.MaskedArray, **kwargs
) -> np.ndarray | np.ma.MaskedArray:
    """Applies a scipy.ndimage filter respecting the mask.

    Parameters
    ----------
    filter : callable
        A scipy.ndimage filter, supporting `mode="constant"`.
    image : np.ndarray | np.ma.MaskedArray
        A numpy.ndarray or MaskedArray. It is converted to float.

    kwargs are passed to filter.

    Returns
    -------
    np.ndarray | np.ma.MaskedArray
        If the input image was a MaskedArray, the output is also a MaskedArray,
        and the mask is shared with the input image.

    Notes
    -----
    Inspired on https://stackoverflow.com/a/36307291,
    which gives the same result as astropy.convolve.
    """
    image = image.astype(float, copy=False)

    if not isinstance(image, np.ma.MaskedArray):
        return filter(image, **kwargs, mode="constant")

    if kwargs.get("output") is not None:
        raise ValueError("Argument output is not respected for MaskedArray.")

    if not image.mask.any():
        out = filter(image.data, **kwargs, mode="constant")
    else:
        out = filter(image.filled(0), **kwargs, mode="constant")
        norm = filter((~image.mask).astype(float), **kwargs, mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(image.mask, np.nan, out / norm)

    return np.ma.MaskedArray(out, image.mask)


def _normalized_gradient(input: np.ndarray) -> list[np.ndarray]:
    """Calculates the normalized gradient of a scalar field.

    Parameters
    ----------
    input : numpy.ndarray
        Input field.

    Returns
    -------
    list of numpy.ndarray
        The normalized gradient of the scalar field.
    """
    grad = np.gradient(input.astype(float, copy=False))

    if input.ndim == 1:
        return [np.sign(grad, out=grad)]

    norm = _euclidean_norm(grad)
    mask = norm > 0
    for x in grad:
        np.divide(x, norm, where=mask, out=x)
    return grad


def smo(
    input: np.ndarray | np.ma.MaskedArray, *, sigma: float, size: int
) -> np.ndarray | np.ma.MaskedArray:
    """Applies the Silver Mountain Operator (SMO) to a scalar field.

    Parameters
    ----------
    input : numpy.ndarray
        Input field.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.

    Returns
    -------
    numpy.ndarray | np.ma.MaskedArray

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical object size.
    """
    input = _filter(gaussian_filter, input, sigma=sigma)
    grad = _normalized_gradient(input)
    grad = [_filter(uniform_filter, x, size=size) for x in grad]
    return _euclidean_norm(grad)


def smo_rv(
    shape: tuple[int, ...], *, sigma: float, size: int, random_state=None
) -> rv_continuous:
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
        By default, numpy.random.default_rng(seed=42).

    Returns
    -------
    scipy.stats.rv_continuous.
    """
    if random_state is None:
        random_state = np.random.default_rng(seed=42)

    image = random_state.uniform(size=shape)
    smo_image = smo(image, sigma=sigma, size=size)
    return _rv(smo_image)
