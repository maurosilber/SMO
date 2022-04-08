import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import common
from .data import pipeline


def image_pipeline(axes: ImageGrid, pipeline: common.SMOPipeline):
    # Intensity
    mappable = axes[0].imshow(
        pipeline.intensity,
        cmap=common.cmap.intensity,
        rasterized=True,
        vmin=1,
        vmax=1000,
    )
    axes[0].cax.colorbar(mappable, label="Intensity", ticks=(1, 1000)).set_ticklabels(
        (1, "$10^3$")
    )

    # Gradient
    mappable = axes[1].imshow(
        pipeline.angle, cmap=common.cmap.angle, vmin=-np.pi, vmax=np.pi, rasterized=True
    )
    axes[1].cax.colorbar(
        mappable, label="Gradient\nAngle", ticks=(-np.pi, np.pi)
    ).set_ticklabels((r"$-\pi$", r"$\pi$"))

    # SMO
    mappable = axes[2].imshow(
        pipeline.smo, cmap=common.cmap.smo, vmin=0, vmax=1, rasterized=True
    )
    axes[2].cax.colorbar(mappable, label="SMO value", ticks=(0.0, 1.0)).set_ticklabels(
        (0.0, 1.0)
    )

    # Remove ticks and align labels
    for ax in axes:
        ax.set(xticks=(), yticks=())
        for text, align in zip(ax.cax.get_yticklabels(), ("baseline", "top")):
            text.set_verticalalignment(align)

        ax.cax.yaxis.label.set_size("small")
        ax.cax.tick_params(labelsize="small")


if __name__ == "__main__":
    fig = plt.figure(figsize=(2, 3))
    axes_images = ImageGrid(
        fig,
        111,  # gs[0].get_position(fig).bounds,
        (3, 1),
        axes_pad=0.05,
        cbar_mode="each",
        cbar_location="left",
    )
    image_pipeline(axes_images, pipeline)
    fig.show()
