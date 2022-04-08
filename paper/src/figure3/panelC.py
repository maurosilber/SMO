import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from .. import common
from .data import color_background, image_true, pipeline


def intensity_background(axes, intensity_thresholds, image, bg_mask):
    bins = np.linspace(-5, 6, 100)

    for ax, (color, t) in zip(axes, intensity_thresholds.items()):
        mask_threshold = image < t

        # Mask
        ax[0].imshow(mask_threshold, cmap="gray", rasterized=True)

        # Histogram
        ax[1].hist(image.flat, bins=bins, alpha=0.5, color="gray")
        ax[1].hist(image[bg_mask], bins=bins, histtype="step", color=color_background)
        ax[1].hist(image[mask_threshold], bins=bins, alpha=0.5, color=color)
        ax[1].set(yscale="log", xlim=(-5, 5))

        common.align_axes(ax[0], ax[1], orientation="y")

        for a in ax.flat:
            a.set(xticks=(), yticks=())
            for spine in a.spines.values():
                spine.set_color(color)


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 2, figsize=(1, 1))

    intensity_thresholds = {
        "red": scipy.stats.norm.ppf(0.5),
        "green": scipy.stats.norm.ppf(0.9),
        "blue": 5,
    }

    intensity_background(
        axes, intensity_thresholds, pipeline.intensity, image_true.mask
    )
    plt.show()
