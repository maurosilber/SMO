import matplotlib.pyplot as plt
import numpy as np

from .. import common
from .data import color_background, image_true, pipeline, smo_rv


def smo_background(axes, intensity_thresholds, pipeline, fg_mask):
    smo_image = pipeline.smo
    bins = np.linspace(-5, 6, 100)

    for ax, (color, t) in zip(axes, intensity_thresholds.items()):
        mask_threshold = smo_image < t

        # Mask
        ax[0].imshow(mask_threshold, cmap="gray", rasterized=True)

        # Histogram
        for a, density in zip(ax[1:], (False, True)):
            a.hist(
                pipeline.intensity[fg_mask],
                bins=bins,
                histtype="step",
                density=density,
                color=color_background,
            )
            a.hist(
                pipeline.intensity[mask_threshold],
                bins=bins,
                alpha=0.5,
                density=density,
                color=color,
            )
            a.set(yscale="log", xlim=(-5, 5))

            common.align_axes(ax[0], a, orientation="y")

        for a in ax.flat:
            a.set(xticks=(), yticks=())
            for spine in a.spines.values():
                spine.set_color(color)


if __name__ == "__main__":
    fig, axes = plt.subplots(3, 3, figsize=(2, 2))
    smo_thresholds = {"blue": 0.75, "green": smo_rv.ppf(0.9), "red": smo_rv.ppf(0.1)}
    smo_background(axes, smo_thresholds, pipeline, image_true.mask)
    plt.show()
