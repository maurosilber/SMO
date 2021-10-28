from typing import Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import numpy as np
from napari.types import ImageData, LabelsData
from napari_plugin_engine import napari_hook_implementation

from ..api import SMO
from ..smo import smo

SizeSlider = Annotated[int, {"widget_type": "Slider", "value": 7, "min": 2, "max": 20}]
SigmaSlider = Annotated[
    float, {"widget_type": "FloatSlider", "value": 0, "min": 0, "max": 10, "step": 0.1}
]
Probability = Annotated[
    float, {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.01}
]

SHAPE = (1024, 1024)
MaskError = ValueError("A boolean mask excluding saturated pixels must be provided.")


def mask(
    image: ImageData, saturation: Annotated[float, {"max": 2 ** 16}]
) -> LabelsData:
    return image >= saturation


def smo_image(
    image: ImageData,
    mask: Optional[LabelsData],
    sigma: SigmaSlider,
    size: SizeSlider,
) -> ImageData:
    if mask is not None:
        image = np.ma.MaskedArray(image, mask)
    return smo(image, sigma=sigma, size=size)


def smo_probability(
    image: ImageData,
    mask: LabelsData,
    sigma: SigmaSlider,
    size: SizeSlider,
) -> ImageData:
    if mask is None:
        raise MaskError

    image = np.ma.MaskedArray(image, mask)
    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.smo_probability(image)


def background_correction(
    image: ImageData,
    mask: LabelsData,
    sigma: SigmaSlider,
    size: SizeSlider,
    threshold: Probability = 0.05,
    background_quantile: Probability = 0.5,
) -> ImageData:
    if mask is None:
        raise MaskError

    image = np.ma.MaskedArray(image, mask)
    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    bg_rv = smo.bg_rv(image, threshold=threshold)
    return image - bg_rv.ppf(background_quantile)


def background_probability(
    image: ImageData,
    mask: LabelsData,
    sigma: SigmaSlider,
    size: SizeSlider,
    threshold: Probability = 0.05,
) -> ImageData:
    if mask is None:
        raise MaskError

    image = np.ma.MaskedArray(image, mask)
    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.bg_probability(image, threshold=threshold)


@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        mask,
        smo_image,
        smo_probability,
        background_correction,
        background_probability,
    ]
