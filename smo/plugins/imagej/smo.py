# @ String (visibility=MESSAGE, value="<html>The SMO provides a robust estimation of the background dsitribution.<br>Consult the documentation at <a>https://github.com/maurosilber/SMO</a></html>", required=false) msg  # noqa: E501
# @ImagePlus (label="Input") image
# @String (label="Output", choices={"Background-corrected image", "SMO image", "Background mask"}, style="radioButtonVertical") output_choice  # noqa: E501
# @Float (label="Smoothing (sigma)", description="Size of the gaussian smoothing kernel.", style="slider,format:0.0", min=0, max=10, stepSize=0.1, value=0, persist=true) sigma  # noqa: E501
# @Integer (label="Averaging window", description="<html>Size of the gradient direction averaging kernel<br>Must be smaller than the foreground features.</html>.", style="slider", min=3, max=21, stepSize=2, value=7, persist=true) size  # noqa: E501
# @Float (label="SMO threshold", description="<html>Percentage of background pixels to include in the background distribution estimation.<br>A higher threshold leads to more foreground pixels being (incorrectly) included.</html>", style="slider,format:0.0", min=0, max=100, stepSize=1, value=5, persist=true) threshold  # noqa: E501
# @Float (label="Background percentile", description="<html>Subtract this percentile of the background distribution to the input image.<br>Note that the resulting image will contain negative values in some background regions,<br>but the background will be centered at 0.</html>", style="slider,format:0.0", min=0, max=100, stepSize=1, value=50, persist=true) percentile  # noqa: E501
# @Boolean (label="Add info to log", value=false) log
# @Boolean (label="Show histogram", description="Show histogram of the background distribution.", value=false) show_histogram  # noqa: E501
# @output ImagePlus output

from ij import IJ, ImagePlus
from ij.gui import HistogramWindow
from ij.measure import Measurements
from ij.plugin import ImageCalculator
from ij.process import FloatProcessor
from java.lang import Double
from java.util import Random


def create_mask(image, low_threshold, high_threshold):
    """
    Parameters
    ----------
    image: FloatProcessor
    low_threshold, high_threshold: float

    Returns
    -------
        BinaryProcessor
    """
    image.setThreshold(low_threshold, high_threshold, image.NO_LUT_UPDATE)
    return image.createMask()


def apply_mask(image, mask):
    """Applies mask in-place to image.

    Parameters
    ----------
    image: ImagePlus
    mask: BinaryProcessor

    Returns
    -------
        None
    """
    ip = image.getProcessor()
    ip.setValue(Double.NaN)
    ip.fill(mask)


def smo(image, sigma, size):
    """
    image: FloatProcessor
    sigma: float
    size: int (odd)
    """
    if sigma > 0:
        image.blurGaussian(sigma)
    # Gradient: y
    grad_y = image.duplicate()
    grad_y.convolve3x3([0, 1, 0, 0, 0, 0, 0, -1, 0])
    # Gradient: x
    grad_x = image
    grad_x.convolve3x3([0, 0, 0, -1, 0, 1, 0, 0, 0])
    # Normalize gradient
    for x in range(image.getWidth()):
        for y in range(image.getHeight()):
            xv = grad_x.getf(x, y)
            yv = grad_y.getf(x, y)
            norm = (xv**2 + yv**2) ** 0.5
            if norm == 0:
                continue
            grad_x.setf(x, y, xv / norm)
            grad_y.setf(x, y, yv / norm)
    # Average gradient direction
    kernel = [1.0 / size**2 for _ in range(size**2)]
    grad_x.convolve(kernel, size, size)
    grad_y.convolve(kernel, size, size)
    # Gradient norm
    grad_x.sqr()
    grad_y.sqr()
    ImageCalculator().run(
        "add", ImagePlus("grad_x", grad_x), ImagePlus("grad_y", grad_y)
    )
    image = grad_x
    image.sqrt()
    return image


def create_random_image():
    """Creates a image of random uniform noise.

    Returns
    -------
        FloatProcessor
    """
    pixels = Random(42).doubles(1024 * 1024).toArray()
    return FloatProcessor(1024, 1024, pixels)


def get_quantile(image, quantile):
    """Compute quantile from image.

    Parameters
    ----------
    image: ImagePlus
    quantile: float

    Returns
    -------
        float
    """
    if quantile == 0.5:
        stats = image.getStatistics(Measurements.MEDIAN)
        return stats.median
    else:
        stats = image.getStatistics()
        fraction = quantile * stats.pixelCount
        cdf = 0
        for i, x in enumerate(stats.histogram()):
            if cdf >= fraction:
                break
            cdf += x
        return i * stats.binSize + stats.histMin


def main(image, sigma, size, threshold, percentile, output_choice, log, show_histogram):
    image = image.getProcessor().convertToFloat().duplicate()

    # Calculate SMO image
    smo_image = image.duplicate()
    smo_image = smo(smo_image, sigma, size)
    if output_choice == "SMO image":
        if log:
            IJ.log(
                ", ".join(
                    (
                        "Output: %s" % output_choice,
                        "Smoothing (sigma): %r" % sigma,
                        "Averaging window: %r" % size,
                    )
                )
            )

        smo_image.setMinAndMax(0.0, 1.0)
        return ImagePlus(output_choice, smo_image)

    # Calculate SMO threshold
    random_image = create_random_image()
    random_smo_image = smo(random_image, sigma, size)
    random_smo_image = ImagePlus("random_smo_image", random_smo_image)
    smo_threshold = get_quantile(random_smo_image, threshold / 100)

    # Threshold SMO image and apply mask
    smo_mask = create_mask(smo_image, smo_threshold, 1.0)
    masked_image = ImagePlus("masked_image", image.duplicate())
    apply_mask(masked_image, smo_mask)
    if output_choice == "Background mask":
        if log:
            IJ.log(
                ", ".join(
                    (
                        "Output: %s" % output_choice,
                        "Smoothing (sigma): %r" % sigma,
                        "Averaging window: %r" % size,
                        "SMO threshold: %r" % threshold,
                    )
                )
            )
        return masked_image

    # Compute background
    background = get_quantile(masked_image, percentile / 100)
    if show_histogram:
        HistogramWindow(masked_image)

    # Correct image
    image.add(-background)
    if output_choice == "Background-corrected image":
        if log:
            IJ.log(
                ", ".join(
                    (
                        "Output: %s" % output_choice,
                        "Smoothing (sigma): %r" % sigma,
                        "Averaging window: %r" % size,
                        "SMO threshold: %r" % threshold,
                        "Background percentile: %r" % percentile,
                        "Background: %r" % background,
                    )
                )
            )

        return ImagePlus(output_choice, image)


if __name__ == "__main__":
    output = main(
        image,  # noqa: F821
        sigma,  # noqa: F821
        size,  # noqa: F821
        threshold,  # noqa: F821
        percentile,  # noqa: F821
        output_choice,  # noqa: F821
        log,  # noqa: F821
        show_histogram,  # noqa: F821
    )
