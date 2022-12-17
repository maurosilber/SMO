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
The default output is the input image with the median background subtracted.
Note that it might contain negative values in the background regions,
but the median background should be centered at 0.

Other useful outputs are:
- Background probability: could be used as input for segmentation algorithms.
- Background mask or SMO probability: could be used as input for fitting a surface in
case of non-homogeneous backgrounds.

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
    bg_mask = "Background mask"
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
- *{Options.bg_mask.value}*: the input image with the mask of selected background
pixels.
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
            doc="Percentile of the SMO null hypothesis distribution to use as "
            "threshold. The default value of 5 means that around 5% of the background "
            "pixels will be used to compute the background distribution. "
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

        smo: _SMO = _SMO(
            sigma=self.sigma.value, size=self.size.value, shape=(1024, 1024)
        )
        if self.output_choice.value == Options.smo_image.value:
            y_data = smo.smo_image(image)
        elif self.output_choice.value == Options.smo_probability.value:
            y_data = smo.smo_probability(image)
        elif self.output_choice.value == Options.bg_mask.value:
            y_data = smo.bg_mask(image, threshold=threshold)
        elif self.output_choice.value == Options.bg_probability.value:
            y_data = smo.bg_probability(image, threshold=threshold)
        elif self.output_choice.value == Options.bg_correction.value:
            y_data = smo.bg_corrected(
                image,
                threshold=threshold,
                statistic=lambda x: np.percentile(x, self.bg_percentile.value),
            )

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
        background = smo.bg_mask(image, threshold=threshold).compressed()
        bg_value = np.percentile(background, self.bg_percentile.value)

        subplot = figure.subplot(0, 0)
        subplot.imshow(image, cmap="plasma")
        subplot.set(title=self.x_name.value)

        ax = figure.subplot(1, 1)
        hist_kwargs = dict(bins="auto", density=True, histtype="step")
        ax.hist(image.compressed(), label="All", **hist_kwargs)
        ax.hist(background, label="Background", **hist_kwargs)
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
        ax.set(title="Background-corrected image")

        ax = figure.subplot(2, 1, **share)
        ax.imshow(smo.bg_probability(image), cmap="viridis")
        ax.set(title="Background probability")


# All code below is copied from commit: c6d54531b9fedd0c3377ae3a926fa86efff25c2c
# See https://github.com/maurosilber/SMO

############
# smo.api.py
############
# from __future__ import annotations

# import numpy as np

# from .background import bg_mask, bg_rv
# from .smo import rv_continuous, smo, smo_rv


class _SMO:
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
        """Returns a background-corrected image by subtracting a value.calculated
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

    return np.sqrt(sum(xi**2 for xi in x))


def _filter(filter: callable, input: np.ma.MaskedArray, **kwargs) -> np.ma.MaskedArray:
    """Applies a scipy.ndimage filter respecting the mask.

    Parameters
    ----------
    filter : callable
        A scipy.ndimage filter, supporting `mode="mirror"`.
    input : numpy.ma.MaskedArray

    Keyword arguments are passed to filter.

    Returns
    -------
    numpy.ma.MaskedArray
    """
    if kwargs.get("output") is not None:
        raise ValueError("Argument output is not respected for MaskedArray.")

    if np.ma.is_masked(input):
        out = filter(input.filled(0), **kwargs, mode="mirror")
        mask = ~filter(~input.mask, **kwargs, mode="mirror")
        return np.ma.MaskedArray(out, mask)
    else:
        out = filter(input.data, **kwargs, mode="mirror")
        mask = input.mask
        return np.ma.MaskedArray(out, mask)


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
    and should be less than the typical foreground object size.
    """
    out = np.ma.MaskedArray(input, dtype=float, copy=False)
    out = _filter(gaussian_filter, out, sigma=sigma)
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
    """Generates a random variable of the null hypothesis for the SMO operator
    for a given sigma and size. The null hypothesis is that pixels are uncorrelated
    and drawn from the same distribution.

    In particular, it uses a uniform distribution, as SMO is non-parametric.

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
