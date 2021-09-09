import numpy as np
from scipy.stats import rv_histogram

from .smo import smo


def smo_mask(masked_image, *, sigma, size, threshold):
    """Returns the mask of (some) background noise.

    Parameters
    ----------
    masked_image : numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.
    threshold : float
        Threshold value [0, 1] for the SMO image.

    Returns
    -------
    mask : numpy.array

    Notesvalue
    -----
    Sigma and size are scale parameters, and should be less than the typical object size.
    """
    if not np.ma.isMaskedArray(masked_image):
        raise TypeError(
            "The first parameter should be a np.ma.MaskedArray. "
            "Use np.ma.masked_greater_equal(image, value_of_saturation)."
        )

    masked_image = np.ma.asarray(masked_image)
    smo_image = smo(masked_image, sigma=sigma, size=size)
    return (smo_image < threshold) & ~masked_image.mask


def bg_rv(masked_image, *, sigma, size, threshold) -> rv_histogram:
    """Returns the distribution of background noise.

    It returns an instance of scipy.stats.rv_histogram.
    Use .median() to get the median value,
    or .ppf(percentile) to calculate any other desired percentile.

    Parameters
    ----------
    masked_image : numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or tuple of int
        Averaging window size.
    threshold : float
        Threshold value [0, 1] for the SMO image.

    Returns
    -------
    scipy.stats.rv_histogram

    Notes
    -----
    Sigma and size are scale parameters, and should be less than the typical object size.
    """
    mask = smo_mask(masked_image, sigma=sigma, size=size, threshold=threshold)
    background = masked_image[mask]
    hist = np.histogram(background, bins="fd")
    return rv_histogram(hist)
