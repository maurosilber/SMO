import numpy as np

from .background import bg_rv, smo_mask
from .smo import rv_continuous, smo, smo_rv


class SMO:
    def __init__(
        self, *, sigma: float, size: int, shape: tuple[int, ...], random_state=None
    ):
        """Wrapper to simplify the use of the Silver Mountain Operator (SMO).

        Parameters
        ----------
        sigma : scalar or sequence of scalars
            Standard deviation for Gaussian kernel.
        size : int or sequence of int
            Averaging window size.
        shape : tuple of ints
            Shape of the random image used to estimate the SMO distribution.
        random_state : numpy.random.Generator
            By default, numpy.random.default_rng(seed=42).

        Notes
        -----
        Sigma and size are scale parameters,
        and should be less than the typical object size.
        """
        self.sigma = sigma
        self.size = size
        self.ndim = len(shape)
        self.smo_rv = smo_rv(shape, sigma=sigma, size=size, random_state=random_state)

    def _check_dim(self, image: np.ndarray):
        """Checks that image has the appropiate dimension.

        It should be checked every time self.smo_rv is called.
        """
        if image.ndim != self.ndim:
            raise ValueError(
                f"Dimension of input image is {image.ndim}, "
                f"while this SMO was constructed for dimension {self.ndim}."
            )

    def smo_image(self, image: np.ndarray) -> np.ndarray:
        """Applies the Silver Mountain Operator (SMO) to a scalar field.

        Parameters
        ----------
        input : numpy.ndarray
            Input field.

        Returns
        -------
        numpy.ndarray
        """
        return smo(image, sigma=self.sigma, size=self.size)

    def smo_mask(
        self, masked_image: np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> np.ndarray:
        """Returns the mask of (some) background noise.

        Parameters
        ----------
        masked_image : numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO image.

        Returns
        -------
        numpy.ndarray of booleans
        """
        self._check_dim(masked_image)
        return smo_mask(
            masked_image,
            sigma=self.sigma,
            size=self.size,
            threshold=self.smo_rv.ppf(threshold),
        )

    def bg_rv(
        self, masked_image: np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> rv_continuous:
        """Returns the distribution of background noise.

        It returns an instance of scipy.stats.rv_histogram.
        Use .median() to get the median value,
        or .ppf(percentile) to calculate any other desired percentile.

        Parameters
        ----------
        masked_image : numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO image.

        Returns
        -------
        scipy.stats.rv_continuous
        """
        self._check_dim(masked_image)
        return bg_rv(
            masked_image,
            sigma=self.sigma,
            size=self.size,
            threshold=self.smo_rv.ppf(threshold),
        )

    def bg_probability(
        self, masked_image: np.ma.MaskedArray, *, threshold: float = 0.05
    ) -> np.ndarray:
        """Returns the probability that each pixel doesn't belong to the background.

        It uses the cumulative density function (CDF) of the background distribution
        to assign a value to each pixel.

        Parameters
        ----------
        masked_image : numpy.ma.MaskedArray
            Image. If there are saturated pixels, they should be masked.
        threshold : float
            Threshold value [0, 1] for the SMO image.

        Returns
        -------
        scipy.stats.rv_continuous
        """
        return self.bg_rv(masked_image, threshold=threshold).cdf(masked_image)
