from __future__ import annotations

import numpy as np

from .background import bg_mask, bg_rv
from .smo import rv_continuous, smo, smo_rv


class SMO:
    def __init__(
        self, *, sigma: float, size: int, shape: tuple[int, ...], random_state=None
    ):
        """Wrapper to simplify the use of the Silver Mountain Operator (SMO).

        The SMO instance keeps track of the distribution of the null hypothesis in its
        `smo_rv` attribute, calculated as a random uniform image of shape given by the
        shape parameter.

        All methods require a MaskedArray as input, with saturated pixels masked.
        If it is not provided as a MaskedArray, a mask is generated for the image
        maximum.

        Parameters
        ----------
        sigma : scalar or sequence of scalars
            Standard deviation for Gaussian kernel.
        size : int or sequence of int
            Averaging window size.
        shape : tuple of ints
            Shape of the random image used to estimate the SMO null hypothesis
            distribution.
        random_state : numpy.random.Generator
            To generate the random image. By default, numpy.random.default_rng(seed=42).

        Notes
        -----
        Sigma and size are scale parameters,
        and should be less than the typical foreground object size.
        """
        self.sigma = sigma
        self.size = size
        self.ndim = len(shape)
        self.smo_rv = smo_rv(shape, sigma=sigma, size=size, random_state=random_state)

    def _check_image(self, image: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Checks that the image has the appropriate dimension and has a mask.

        If image is not masked, the maximum intensity values are masked.
        """
        if image.ndim != self.ndim:
            raise ValueError(
                f"Dimension of input image is {image.ndim}, "
                f"while this SMO was constructed for dimension {self.ndim}."
            )

        if isinstance(image, np.ma.MaskedArray):
            return image
        else:
            saturation = image.max()
            return np.ma.masked_greater_equal(image, saturation)

    def smo_image(self, image: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
        """Applies the Silver Mountain Operator (SMO) to a scalar field.

        Parameters
        ----------
        input : numpy.ndarray | np.ma.MaskedArray
            Input field.

        Returns
        -------
        np.ma.MaskedArray
        """
        image = self._check_image(image)
        return smo(image, sigma=self.sigma, size=self.size)

    def smo_probability(
        self, image: np.ndarray | np.ma.MaskedArray
    ) -> np.ma.MaskedArray:
        """Applies the Silver Mountain Operator (SMO) to a scalar field, and
        rescales it with the null hypothesis distribution (self.smo_rv).

        Each pixel has the probability of not belonging to the background,
        according to SMO.

        Parameters
        ----------
        input : numpy.ndarray | np.ma.MaskedArray
            Input field. If there are saturated pixels, they should be masked.

        Returns
        -------
        np.ma.MaskedArray
        """
        image = self.smo_image(image)
        prob = self.smo_rv.cdf(image.data)
        return np.ma.MaskedArray(prob, image.mask)

    def bg_mask(
        self, image: np.ndarray | np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> np.ma.MaskedArray:
        """Returns the input image with only SMO-chosen background pixels unmasked.

        As it is a statistical test, some foreground pixels might be included.

        Parameters
        ----------
        image : numpy.ndarray | numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO probability image.

        Returns
        -------
        numpy.ma.MaskedArray
        """
        image = self._check_image(image)
        return bg_mask(
            image,
            sigma=self.sigma,
            size=self.size,
            threshold=self.smo_rv.ppf(threshold),
        )

    def bg_corrected(
        self,
        image: np.ndarray | np.ma.MaskedArray,
        *,
        statistic: callable = np.median,
        threshold: float = 0.05,
    ) -> np.ma.MaskedArray:
        """Returns a background-corrected image by subtracting a value calculated
        by applying statistic to the sample of background pixels.

        Parameters
        ----------
        image : numpy.ndarray | numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        statistic : callable
            Computes the value to subtract. It receives as input a 1D array of
            background pixels.
        threshold : float
            Threshold value [0, 1] for the SMO probability image.

        Returns
        -------
        np.ma.MaskedArray
            If the input has a mask, it is shared by the output.

        Notes
        -----
        The resulting image might contain negative values in the background regions,
        but the background should be centered at 0.
        """
        image = self._check_image(image)
        bg = self.bg_mask(image, threshold=threshold)
        return image - statistic(bg.compressed())

    def bg_rv(
        self, image: np.ndarray | np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> rv_continuous:
        """Returns the distribution of background intensity.

        It returns an instance of scipy.stats.rv_histogram.
        Use .median() to get the median value,
        or .ppf(percentile) to calculate any other desired percentile.

        As some foreground pixels might be wrongly included, it is not recommended to
        trust percentiles near to 100.

        Parameters
        ----------
        image : numpy.ndarray | numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO probability image.

        Returns
        -------
        scipy.stats.rv_continuous
        """
        image = self._check_image(image)
        return bg_rv(
            image,
            sigma=self.sigma,
            size=self.size,
            threshold=self.smo_rv.ppf(threshold),
        )

    def bg_probability(
        self, image: np.ndarray | np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> np.ma.MaskedArray:
        """Returns the probability that each pixel doesn't belong to the background.

        It uses the cumulative density function (CDF) of the background distribution
        to assign a value to each pixel.

        Parameters
        ----------
        image : numpy.ndarray | numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO probability image.

        Returns
        -------
        np.ma.MaskedArray
            If the input has a mask, it is shared by the output.
        """
        image = self._check_image(image)
        prob = self.bg_rv(image, threshold=threshold).cdf(image.data)
        return np.ma.MaskedArray(prob, image.mask)
