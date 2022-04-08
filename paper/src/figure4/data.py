import skimage.filters
import skimage.restoration
from skimage.io import imread

import smo

from .. import common

# Parameters
sigma, size = 1, 7

# Image
image = imread("data/image.tif").astype(float)
image = image[:300, 200:]  # crop image
pipeline = common.SMOPipeline(image, sigma=sigma, window=size)

smo_rv = smo.smo_rv((1024, 1024), sigma=sigma, size=size)
smo_threshold = smo_rv.ppf(0.05)
smo_mask = pipeline.smo < smo_threshold

# Manual selection of background
x0, x1, y0, y1 = 10, 80, 80, 130
bg_manual = slice(y0, y1), slice(x0, x1)

# Otsu threshold
otsu_threshold = skimage.filters.threshold_otsu(image.astype(int))
otsu_mask = image < otsu_threshold

# Rolling ball
rolling_ball = skimage.restoration.rolling_ball(image, radius=100)

# Backgrounds
bgs = {
    "SMO": image[smo_mask],
    "Manual": image[bg_manual].ravel(),
    "Otsu": image[image < otsu_threshold],
    "Rolling Ball": rolling_ball,
}
