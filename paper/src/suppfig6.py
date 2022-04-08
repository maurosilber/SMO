import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.io import imread

import smo

from . import common

# Parameters
sigma, size = 1, 7

# SMO threshold
smo_rv = smo.smo_rv((1024, 1024), sigma=sigma, size=size)
smo_threshold = smo_rv.ppf(0.05)

# Image
image = imread("data/image.tif")
image = image[:300, 200:]  # crop image

# SMO image and mask
smo_image = smo.smo(image, sigma=sigma, size=size)
smo_mask = smo_image < smo_threshold

# Manual selection of background
x0, x1, y0, y1 = 0, 80, 80, 130
bg_manual = slice(y0, y1), slice(x0, x1)

s = 20
smo_mask_improved = morphology.binary_dilation(smo_mask, morphology.square(s))
smo_mask_improved = morphology.binary_erosion(
    smo_mask_improved, morphology.square(2 * s)
)
smo_mask_improved = smo_mask_improved & smo_mask

bgs = pd.Series(
    {
        "Manual": image[bg_manual].ravel(),
        "SMO": image[smo_mask],
        "SMO + morph": image[smo_mask_improved],
    }
).to_frame("data")

bgs["Mean"] = bgs.data.apply(np.mean)
bgs["Mean_error"] = bgs.data.apply(
    lambda x: np.random.choice(x.ravel(), size=[1000, x.size // 100]).mean(1).std(), 1
)

masks = {"SMO": smo_mask, "SMO + morph": smo_mask_improved}

fig = plt.figure()
left, right = fig.add_gridspec(1, 2, width_ratios=(1, 3))
axes_masks = left.subgridspec(2, 1).subplots()
axes_hist = fig.add_subplot(right)

for ax, (name, mask) in zip(axes_masks, masks.items()):
    ax.imshow(mask, rasterized=True)
    ax.set(xticks=(), yticks=(), ylabel=name)

for name, bg in bgs.iterrows():
    axes_hist.hist(
        bg.data,
        bins=np.arange(195, 250),
        histtype="step",
        density=True,
        label=f"{name} $({bg.Mean:.1f} \\pm {bg.Mean_error:.1f})$",
    )
axes_hist.set(xlabel="Intensity [a. u.]", yscale="log")
axes_hist.legend(title="Method (mean)")

common.add_subplot_label(axes_masks, "A")
common.add_subplot_label(axes_hist, "B")
common.save_or_show(fig, "suppfig6")
