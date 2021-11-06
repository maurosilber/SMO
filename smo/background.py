import numpy as np

from .smo import _rv, rv_continuous, smo


def bg_mask(
    masked_image: np.ma.MaskedArray, *, sigma: float, size: int, threshold: float
) -> np.ma.MaskedArray:
    """Returns the input image with only SMO-chosen background pixels unmasked.

    As it is a statistical test, some foreground pixels might be included.

    Parameters
    ----------
    masked_image : numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.
    threshold : float
        Threshold value [0, 1] for the SMO image. Use `smo.smo.smo_rv` to select
        a proper value, such as `smo_rv.ppf(0.05)`.

    Returns
    -------
    numpy.ma.MaskedArray

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical foreground object size.
    """
    if not np.ma.isMaskedArray(masked_image):
        raise TypeError(
            "The first parameter should be a np.ma.MaskedArray. "
            "Use np.ma.masked_greater_equal(image, value_of_saturation)."
        )

    smo_image = smo(masked_image, sigma=sigma, size=size)
    mask = (smo_image > threshold).filled(True)
    return np.ma.MaskedArray(masked_image, mask)


def bg_rv(
    masked_image: np.ma.MaskedArray, *, sigma: float, size: int, threshold: float
) -> rv_continuous:
    """Returns the distribution of background intensity.

    It returns an instance of scipy.stats.rv_histogram.
    Use .median() to get the median value,
    or .ppf(percentile) to calculate any other desired percentile.

    As some foreground pixels might be wrongly included, it is not recommended to
    trust percentiles near to 100.

    Parameters
    ----------
    masked_image : numpy.ma.MaskedArray
        Image. If there are saturated pixels, they should be masked.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.
    threshold : float
        Threshold value [0, 1] for the SMO image. Use `smo.smo.smo_rv` to select
        a proper value, such as `smo_rv.ppf(0.05)`.

    Returns
    -------
    scipy.stats.rv_continuous

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical foreground object size.
    """
    background = bg_mask(masked_image, sigma=sigma, size=size, threshold=threshold)
    return _rv(background.compressed())
