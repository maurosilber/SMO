from __future__ import annotations

from enum import Enum

import numpy as np
from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.text import Float, Integer
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import rv_continuous, rv_histogram

__doc__ = """\
SMO
===

The Silver Mountain Operator **(SMO)** estimates the background intensity distribution
by exploiting the lack of local correlation in background regions.

The default choice is to subtract the median background to the input image.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

Note: the SMO supports ND images, but this plugin is only implemented for 2D images.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^
Homogeneous background: the image is expected to have a background intensity
distribution that does not have a spatial dependency. For instance, a gaussian
distribution with the same mean and variance for every point in space. Note that
the background distribution does not need to be gaussian.

Non-homogeneous background: if the background is not homogeneous, the SMO
probability or mask could be useful as outputs for other plugins, that fit a
surface to the input image with weights given by the SMO probability or mask.

The input is expected to be a single-channel 2D image, where the foreground is
brighter than the background. The pixels which are saturated must be masked. For
convenience, a setting to mask them is provided as part of this module.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^
The default output is a the input image with the median background subtracted.

Other useful outputs are:
- Background probability: could be used as input for segmentation algorithms.
- SMO probability or mask: could be used as input for fitting a surface in case
of non-homogeneous backgrounds.

Technical notes
^^^^^^^^^^^^^^^
A smo_rv distribution is recomputed for each image, but it could be reused for all
images with the same sigma and size parameters. The overhead should be negligible.

References
^^^^^^^^^^
- Silberberg M., Grecco H. E., 2021 (to be published).
- GitHub repository: (`link <https://github.com/maurosilber/SMO>`__)
"""


class Options(Enum):
    smo_image = "SMO image"
    smo_probability = "SMO probability"
    smo_mask = "SMO mask"
    bg_probability = "Background probability"
    bg_correction = "Background correction"


class SMO(ImageProcessing):
    module_name = "SMO"
    variable_revision_number = 1

    def create_settings(self):
        # from superclass (ImageProcessing):
        # -  x_name: input
        # -  y_name: output
        super(SMO, self).create_settings()

        self.output_choice = Choice(
            text="Output choice:",
            choices=[o.value for o in Options],
            value=Options.bg_correction.value,  # Default
            doc=f"""Choose what to calculate:

- *{Options.smo_image.value}*: the SMO image, which measures the local correlation of
the gradient direction.
- *{Options.smo_probability.value}*: the SMO image rescaled with the CDF of the SMO
distribution for an uncorrelated background only image.
- *{Options.smo_mask.value}*: the SMO mask used to select background pixels from the
input image.
- *{Options.bg_probability.value}*: the input image rescaled with the CDF of the
computed background distribution.
- *{Options.bg_correction.value}*: the input image with the chosen background intensity
percentile subtracted.
""",
        )
        self.saturation = Float(
            text="Saturation intensity",
            value="nan",
            doc="Pixels with intensity greater or equal than this value are masked."
            " It is important to mask saturated areas, otherwise the algorithm would"
            " recognize them as background."
            "\n"
            "The default value 'nan', uses the image maximum."
            "\n"
            "CellProfiler rescales the image to a 0-1 range based on the metadata."
            " Hence, 1 could be an appropriate value. But, if the metadata is missing,"
            " it uses the maximum value allowable by the data type, e.g. 2^16 for"
            " 16-bit images, even if the underlying data saturates at a lower value.",
        )
        self.sigma = Float(
            text="Smoothing scale",
            value=0,
            minval=0,
            doc="A scaling factor that supplies the sigma for a gaussian"
            " smoothing. It might be useful to diminish the noise in foreground"
            " structures such that the gradient is properly computed."
            " Important: while the smoothed image is used for finding background"
            " pixels, the pixels are extracted from the original non-smoothed image.",
        )
        self.size = Integer(
            text="Averaging window",
            value=7,
            minval=2,
            doc="Width of the window where the image gradient direction is averaged."
            " It should be smaller than the typical cell size.",
        )
        self.smo_threshold = Float(
            text="SMO percentile threshold",
            value=5,
            minval=0,
            maxval=100,
            doc="Percentile of the SMO distribution to use as threshold. "
            "The default value of 5 means that around 5% of the background pixels"
            " will be used to compute the background distribution. "
            "A higher threshold can fail to exclude non-background pixels,"
            " while a lower threshold can fail to include enough pixels"
            " for an accurate estimation.",
        )
        self.bg_percentile = Float(
            text="Background percentile",
            value=50,
            minval=0,
            maxval=100,
            doc="Percentile of the background distribution to calculate"
            " and subtract to the image. By default, the median: 50.",
        )

    def settings(self):
        settings = super(SMO, self).settings()
        return settings + [
            self.output_choice,
            self.saturation,
            self.sigma,
            self.size,
            self.smo_threshold,
            self.bg_percentile,
        ]

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        x = workspace.image_set.get_image(self.x_name.value)

        saturation = self.saturation.value
        if np.isnan(saturation):
            saturation = x.image.max()

        image = np.ma.MaskedArray(x.image, ~x.mask)
        image = np.ma.masked_greater_equal(image, saturation)

        threshold = self.smo_threshold.value / 100
        bg_quantile = self.bg_percentile.value / 100

        smo = _SMO(sigma=self.sigma.value, size=self.size.value, shape=(1024, 1024))
        if self.output_choice.value == Options.smo_image.value:
            y_data = smo.smo_image(image)
        elif self.output_choice.value == Options.smo_probability.value:
            y_data = smo.smo_probability(image)
        elif self.output_choice.value == Options.smo_mask.value:
            y_data = smo.smo_mask(image, threshold=threshold)
        elif self.output_choice.value == Options.bg_probability.value:
            y_data = smo.bg_probability(image, threshold=threshold)
        elif self.output_choice.value == Options.bg_correction.value:
            bg_rv = smo.bg_rv(image, threshold=threshold)
            y_data = image - bg_rv.ppf(bg_quantile)

        y = Image(
            dimensions=x.dimensions,
            image=y_data.data,
            mask=y_data.mask,
            parent_image=x,
            convert=False,
        )
        workspace.image_set.add(self.y_name.value, y)

        if self.show_window:
            workspace.display_data.x_data = image
            workspace.display_data.dimensions = x.dimensions
            workspace.display_data.smo = smo

    def volumetric(self):
        return True

    def display(self, workspace, figure, cmap=None):
        layout = (3, 2)

        figure.set_subplots(
            dimensions=workspace.display_data.dimensions, subplots=layout
        )

        image: np.ma.MaskedArray = workspace.display_data.x_data
        smo: _SMO = workspace.display_data.smo
        threshold = self.smo_threshold.value / 100
        bg_rv = smo.bg_rv(image, threshold=threshold)
        bg_value = bg_rv.ppf(self.bg_percentile.value / 100)

        subplot = figure.subplot(0, 0)
        subplot.imshow(image, cmap="plasma")
        subplot.set(title=self.x_name.value)

        ax = figure.subplot(1, 1)
        hist_kwargs = dict(bins="auto", density=True, histtype="step")
        ax.hist(image.compressed(), label="All", **hist_kwargs)
        ax.hist(
            image[smo.smo_mask(image, threshold=threshold)],
            label="Background",
            **hist_kwargs,
        )
        ax.axvline(bg_value, label=f"{self.bg_percentile.value} percentile", color="k")
        ax.legend()
        ax.set(title="Intensity histogram", yscale="log")

        share = {"sharex": subplot, "sharey": subplot}
        ax = figure.subplot(1, 0, **share)
        ax.imshow(smo.smo_image(image), cmap="viridis")
        ax.set(title="SMO image")

        ax = figure.subplot(2, 0, **share)
        ax.imshow(smo.smo_probability(image), cmap="viridis")
        ax.set(title="SMO probability")

        ax = figure.subplot(0, 1, **share)
        ax.imshow(image - bg_value, cmap="plasma")
        ax.set(title="Background corrected image")

        ax = figure.subplot(2, 1, **share)
        ax.imshow(smo.bg_probability(image), cmap="viridis")
        ax.set(title="Background probability")


# All code below is copied from commit: 5ab0f3e4a13a304b38886e5322e64ecd94ddfca3
# See https://github.com/maurosilber/SMO

############
# smo.api.py
############
# from __future__ import annotations

# import numpy as np

# from .background import bg_rv, smo_mask
# from .smo import rv_continuous, smo, smo_rv


class _SMO:
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
        """Checks that image has the appropriate dimension.

        It should be checked every time self.smo_rv is called.
        """
        if image.ndim != self.ndim:
            raise ValueError(
                f"Dimension of input image is {image.ndim}, "
                f"while this SMO was constructed for dimension {self.ndim}."
            )

    def smo_image(
        self, image: np.ndarray | np.ma.MaskedArray
    ) -> np.ndarray | np.ma.MaskedArray:
        """Applies the Silver Mountain Operator (SMO) to a scalar field.

        Parameters
        ----------
        input : numpy.ndarray | np.ma.MaskedArray
            Input field.

        Returns
        -------
        numpy.ndarray | np.ma.MaskedArray
        """
        return smo(image, sigma=self.sigma, size=self.size)

    def smo_probability(
        self, image: np.ndarray | np.ma.MaskedArray
    ) -> np.ndarray | np.ma.MaskedArray:
        """Applies the Silver Mountain Operator (SMO) to a scalar field.

        Parameters
        ----------
        input : numpy.array | np.ma.MaskedArray
            Input field.

        Returns
        -------
        numpy.array | np.ma.MaskedArray
        """
        self._check_dim(image)
        smo_prob = self.smo_rv.cdf(self.smo_image(image))
        if isinstance(image, np.ma.MaskedArray):
            smo_prob = np.ma.MaskedArray(smo_prob, image.mask)
        return smo_prob

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
    ) -> np.ma.MaskedArray:
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
        np.ma.MaskedArray
            The output shares the input's mask.
        """
        out = self.bg_rv(masked_image, threshold=threshold).cdf(masked_image)
        return np.ma.MaskedArray(out, masked_image.mask)


############
# smo.smo.py
############
# from __future__ import annotations

# import numpy as np
# from scipy.ndimage import gaussian_filter, uniform_filter
# from scipy.stats import rv_continuous, rv_histogram


def _rv(x: np.ndarray) -> rv_continuous:
    hist = np.histogram(x.flat, bins="auto")
    return rv_histogram(hist)


def _euclidean_norm(x: list[np.ndarray]) -> np.ndarray:
    if len(x) == 1:
        return np.abs(x[0])

    return np.sqrt(sum(xi ** 2 for xi in x))


def _filter(
    filter: callable, input: np.ndarray | np.ma.MaskedArray, **kwargs
) -> np.ma.MaskedArray:
    """Applies a scipy.ndimage filter respecting the mask.

    Parameters
    ----------
    filter : callable
        A scipy.ndimage filter, supporting `mode="constant"`.
    input : numpy.ndarray | numpy.ma.MaskedArray
        If it is not a MaskedArray, it is converted to a MaskedArray.

    Keyword arguments are passed to filter.

    Returns
    -------
    numpy.ma.MaskedArray
        The mask is shared with the input image.

    Notes
    -----
    Inspired on https://stackoverflow.com/a/36307291,
    which gives the same result as astropy.convolve.
    """
    if kwargs.get("output") is not None:
        raise ValueError("Argument output is not respected for MaskedArray.")

    # mask=None creates a np.zeros_like(input.data, bool) if no mask is provided.
    input = np.ma.MaskedArray(input, mask=None)

    out = filter(input.filled(0), **kwargs, mode="constant")
    norm = filter((~input.mask).astype(float), **kwargs, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(input.mask, np.nan, out / norm)

    return np.ma.MaskedArray(out, input.mask)


def _normalized_gradient(input: np.ma.MaskedArray) -> list[np.ma.MaskedArray]:
    """Calculates the normalized gradient of a scalar field.

    Parameters
    ----------
    input : numpy.ma.MaskedArray
        Input field.

    Returns
    -------
    list of numpy.ma.MaskedArray
        The normalized gradient of the scalar field.
    """
    grad = np.gradient(input)

    if input.ndim == 1:
        return [np.sign(grad, out=grad)]

    norm = np.ma.MaskedArray(_euclidean_norm(grad), mask=input.mask)
    for x in grad:
        np.divide(x.data, norm.data, where=(norm > 0).filled(False), out=x.data)

    return [np.ma.MaskedArray(x.data, norm.mask) for x in grad]


def smo(
    input: np.ndarray | np.ma.MaskedArray, *, sigma: float, size: int
) -> np.ndarray | np.ma.MaskedArray:
    """Applies the Silver Mountain Operator (SMO) to a scalar field.

    Parameters
    ----------
    input : numpy.ndarray | numpy.ma.MaskedArray
        Input field.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel.
    size : int or sequence of int
        Averaging window size.

    Returns
    -------
    numpy.ndarray | numpy.ma.MaskedArray

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical object size.
    """
    input = input.astype(float, copy=False)
    out = _filter(gaussian_filter, input, sigma=sigma)
    out = _normalized_gradient(out)
    out = [_filter(uniform_filter, x, size=size) for x in out]
    out = _euclidean_norm(out)

    if isinstance(input, np.ma.MaskedArray):
        return out
    else:
        return out.data


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


###################
# smo.background.py
###################
# import numpy as np

# from .smo import _rv, rv_continuous, smo


def smo_mask(
    masked_image: np.ma.MaskedArray, *, sigma: float, size: int, threshold: float
) -> np.ndarray:
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
    numpy.ndarray of booleans

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical object size.
    """
    if not np.ma.isMaskedArray(masked_image):
        raise TypeError(
            "The first parameter should be a np.ma.MaskedArray. "
            "Use np.ma.masked_greater_equal(image, value_of_saturation)."
        )

    smo_image = smo(masked_image, sigma=sigma, size=size)
    return (smo_image < threshold) & ~masked_image.mask


def bg_rv(
    masked_image: np.ma.MaskedArray, *, sigma: float, size: int, threshold: float
) -> rv_continuous:
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
    size : int or sequence of int
        Averaging window size.
    threshold : float
        Threshold value [0, 1] for the SMO image.

    Returns
    -------
    scipy.stats.rv_continuous

    Notes
    -----
    Sigma and size are scale parameters,
    and should be less than the typical object size.
    """
    mask = smo_mask(masked_image, sigma=sigma, size=size, threshold=threshold)
    background_values = masked_image[mask].compressed()
    return _rv(background_values)
