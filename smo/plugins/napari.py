from typing import Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import numpy as np
from napari.types import ImageData, LabelsData
from napari_plugin_engine import napari_hook_implementation

from .. import SMO

SizeSlider = Annotated[int, {"widget_type": "Slider", "value": 7, "min": 2, "max": 20}]
SigmaSlider = Annotated[
    float, {"widget_type": "FloatSlider", "value": 0, "min": 0, "max": 10, "step": 0.1}
]
Probability = Annotated[
    float, {"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.01}
]

SHAPE = (1024, 1024)


def smo_image(
    image: ImageData,
    mask: Optional[LabelsData],
    sigma: SigmaSlider,
    size: SizeSlider,
) -> ImageData:
    if mask is not None:
        image = np.ma.MaskedArray(image, mask)

    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.smo_image(image)


def smo_probability(
    image: ImageData,
    mask: Optional[LabelsData],
    sigma: SigmaSlider,
    size: SizeSlider,
) -> ImageData:
    if mask is not None:
        image = np.ma.MaskedArray(image, mask)

    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.smo_probability(image)


def background_correction(
    image: ImageData,
    mask: Optional[LabelsData],
    sigma: SigmaSlider,
    size: SizeSlider,
    threshold: Probability = 0.05,
    background_quantile: Probability = 0.5,
) -> ImageData:
    if mask is not None:
        image = np.ma.MaskedArray(image, mask)

    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.bg_corrected(
        image,
        threshold=threshold,
        statistic=lambda x: np.quantile(x, background_quantile),
    )


def background_probability(
    image: ImageData,
    mask: Optional[LabelsData],
    sigma: SigmaSlider,
    size: SizeSlider,
    threshold: Probability = 0.05,
) -> ImageData:
    if mask is not None:
        image = np.ma.MaskedArray(image, mask)

    smo = SMO(sigma=sigma, size=size, shape=SHAPE)
    return smo.bg_probability(image, threshold=threshold)


@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        smo_image,
        smo_probability,
        background_correction,
        background_probability,
    ]
