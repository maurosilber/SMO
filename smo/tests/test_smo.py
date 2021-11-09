import numpy as np

from smo import SMO
from smo.smo import rv_continuous


def test_smo_api():
    smo = SMO(sigma=1, size=7, shape=(1024, 1024))

    assert isinstance(smo.smo_rv, rv_continuous)

    np.random.seed(42)
    unmasked_image = np.random.uniform(size=(1024, 1024))
    masked_image = np.ma.MaskedArray(unmasked_image)

    for image in (unmasked_image, masked_image):
        assert np.ma.isMaskedArray(smo.smo_image(image))
        assert np.ma.isMaskedArray(smo.smo_probability(image))
        assert np.ma.isMaskedArray(smo.bg_mask(image))
        assert np.ma.isMaskedArray(smo.bg_corrected(image))
        assert isinstance(smo.bg_rv(image), rv_continuous)
        assert np.ma.isMaskedArray(smo.bg_probability(image))
