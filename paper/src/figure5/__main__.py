import matplotlib.pyplot as plt
import numpy as np

from .. import common
from ..BBBC025 import methods, process, segmentation
from ..common import ecdf
from .data import files

segmentation_thresholds = (1400, 1200)
all_methods = methods.get_all_methods()


fig, axes = plt.subplots(1, len(files), sharex=True, sharey=True)
fig.set_figheight(2)

for ax, file in zip(axes, files):
    image = process.load_image(file)

    mask = segmentation.segment(image, *segmentation_thresholds, disk_size=8)
    bg_manual = image[~mask]
    fg_area = mask.sum() / mask.size

    # Image
    ax_image = ax.inset_axes([0.56, 0.02, 0.4, 0.4])
    ax_image.imshow(np.ma.masked_array(image, ~mask), cmap="gray", rasterized=True)
    ax_image.imshow(np.ma.masked_array(image, mask), rasterized=True)
    ax_image.set(xticks=(), yticks=())
    ax_image.text(-10, 0, f"{fg_area:.0%}", ha="right", fontsize="large")

    # CDFs
    ax.plot(*ecdf(bg_manual), label="Manual", color="k", linewidth=3)

    for name, method in all_methods.items():
        mask = method(image)
        bg = image[mask]
        ax.plot(*ecdf(bg), label=name)

    ax.set(xlabel="Intensity")

axes[0].set(ylabel="CDF")
ax.set(xlim=(1100, 1700))

# Legend
ax.legend(
    title="Methods",
    loc="upper left",
    bbox_to_anchor=(1.05, 1),
    handlelength=1,
)

common.save_or_show(fig, "figure5")
