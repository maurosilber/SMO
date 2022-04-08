import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import common


def image_pipeline(axes: ImageGrid, pipeline: common.SMOPipeline, *, slice=None):
    # Intensity
    mappable = axes[0].imshow(
        pipeline.intensity[slice],
        cmap=common.cmap.intensity,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    axes[0].cax.colorbar(mappable, label="Intensity", ticks=(0.0, 1.0)).set_ticklabels(
        (0.0, 1.0)
    )

    # Gradient
    mappable = common.gradient_quiver_plot(axes[1], pipeline.gradient[slice], step=5)
    axes[1].cax.colorbar(mappable, label="Angle", ticks=(-np.pi, np.pi)).set_ticklabels(
        (r"$-\pi$", r"$\pi$")
    )

    # SMO
    mappable = axes[2].imshow(
        pipeline.smo[slice], cmap=common.cmap.smo, vmin=0.0, vmax=0.4, rasterized=True
    )
    axes[2].cax.colorbar(mappable, label="SMO value", ticks=(0.0, 0.4))

    # Remove ticks and align labels
    for ax in axes:
        ax.set(xticks=(), yticks=())
        for text, align in zip(ax.cax.get_yticklabels(), ("baseline", "top")):
            text.set_verticalalignment(align)


if __name__ == "__main__":
    from .data import pipeline

    fig = plt.figure()
    axes_images = ImageGrid(
        fig,
        111,
        (3, 1),
        axes_pad=0.05,
        cbar_mode="each",
        cbar_location="left",
    )
    image_pipeline(axes_images, pipeline, slice=(slice(0, 64), slice(0, 64)))
    plt.show()
