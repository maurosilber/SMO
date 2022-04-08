import numpy as np
from skimage.morphology import binary_erosion, disk


def get_all_methods():
    return {
        "SMO": get_smo(),
        "triangle": get_intensity_thresholding("triangle"),
        "li": get_intensity_thresholding("li"),
        "isodata": get_intensity_thresholding("isodata"),
        "otsu": get_intensity_thresholding("otsu"),
        "yen": get_intensity_thresholding("yen"),
        "stardist": get_stardist(),
        "cellpose": get_cellpose(),
    }


def erode_background_mask(mask, dilation: int = 10):
    return binary_erosion(mask, disk(dilation))


##########################
# Intensity thresholding #
##########################

thresholding_methods = ("triangle", "li", "isodata", "otsu", "yen")


def get_intensity_thresholding(name):
    import skimage.filters.thresholding

    method = getattr(skimage.filters.thresholding, f"threshold_{name}")

    def intensity_thresholding(image):
        return image < method(image)

    return intensity_thresholding


############
# CellPose #
############


def get_cellpose():
    from cellpose.core import use_gpu
    from cellpose.models import Cellpose

    cellpose_model = Cellpose(gpu=use_gpu(), model_type="cyto")

    def cellpose(image):
        # channels = [cytoplasm, nucleus]
        #   grayscale=0, R=1, G=2, B=3
        channels = [0, 0]
        labels = cellpose_model.eval([image], diameter=None, channels=channels)[0][0]
        mask = labels == 0
        return erode_background_mask(mask)

    return cellpose


############
# Stardist #
############


def get_stardist():
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    stardist_model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def stardist(image):
        labels, _ = stardist_model.predict_instances(normalize(image))
        mask = labels == 0
        return erode_background_mask(mask)

    return stardist


#######
# SMO #
#######


def get_smo():
    from smo.background import smo_mask
    from smo.smo import smo_rv

    sigma, size = 0, 7
    smo_threshold = smo_rv((1024, 1024), sigma=sigma, size=size).ppf(0.05)

    def smo(image):
        saturation = 2 ** 16
        masked_image = np.ma.masked_greater_equal(image, saturation - 8)
        return smo_mask(masked_image, sigma=sigma, size=size, threshold=smo_threshold)

    return smo
