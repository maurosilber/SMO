import numpy as np
from pytest import raises

from smo.api import SMO
from smo.smo import rv_continuous


def test_smo_api():
    smo = SMO(sigma=1, size=7, shape=(1024, 1024))
    image = np.zeros((1024, 1024))
    masked_image = np.ma.MaskedArray(image)

    assert smo.smo_image(image).shape == image.shape
    assert isinstance(smo.smo_rv, rv_continuous)

    with raises(TypeError):
        smo.bg_rv(image)

    assert isinstance(smo.bg_rv(masked_image), rv_continuous)
    assert smo.bg_probability(masked_image).shape == masked_image.shape
