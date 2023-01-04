import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.morphology import binary_dilation, square

from .. import common


def masks(axes, image, otsu_mask, smo_mask, rolling_ball):
    axes[0].imshow(
        np.ma.masked_array(image, ~otsu_mask),
        cmap=common.cmap.intensity,
        rasterized=True,
    )
    axes[0].imshow(np.ma.masked_array(image, otsu_mask), cmap="gray", rasterized=True)

    smo_mask = binary_dilation(smo_mask, square(2))  # Dilated for illustration
    bg_smo = np.ma.masked_array(image, ~smo_mask)
    axes[1].imshow(
        bg_smo,
        cmap=common.cmap.intensity,
        vmax=np.quantile(bg_smo.compressed(), 0.95),
        rasterized=True,
    )
    axes[1].imshow(np.ma.masked_array(image, smo_mask), cmap="gray", rasterized=True)

    axes[2].imshow(rolling_ball, cmap=common.cmap.intensity, vmax=210, rasterized=True)

    axes[1].set(title="Backgrounds")

    for ax, title in zip(axes, ("Otsu", "SMO", "Rolling\nball")):
        ax.set_xlabel(title, fontsize="small")

    for ax in axes:
        ax.set(xticks=(), yticks=())


if __name__ == "__main__":
    from .data import image, otsu_mask, rolling_ball, smo_mask

    fig = plt.figure(figsize=(2, 2))
    axes = ImageGrid(fig, 111, (1, 3), axes_pad=0.05)
    masks(axes, image, otsu_mask, smo_mask, rolling_ball)
    fig.show()
