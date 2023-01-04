import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import common
from .data import (
    cmap_int_thres,
    cmap_smo_thres,
    color_background,
    color_foreground,
    image_true,
    pipeline,
    smo_rv,
)
from .panelA import image_pipeline
from .panelB import hist2d
from .panelC import intensity_background
from .panelD import smo_background

fig = plt.figure()
fig.set_figheight(2.5)

gs_A, gs_B, gs_right = fig.add_gridspec(1, 3, wspace=0.3, width_ratios=(1, 1, 1.5))

# Panel A: image
axes_images = ImageGrid(
    fig,
    gs_A.get_position(fig).bounds,
    (3, 1),
    axes_pad=0.05,
    cbar_mode="each",
    cbar_location="left",
)
image_pipeline(axes_images, pipeline)

# Panel B: histograms
axes_joint = fig.add_subplot(gs_B.subgridspec(2, 1)[0])
axes_2d = common.MarginalAxes(axes_joint)
hist2d(axes_2d, pipeline, image_true.mask)

# Panel C
gs_right = gs_right.subgridspec(7, 4, width_ratios=(0.5, 1, 2, 2))
axes_int = gs_right[:3, 1:3].subgridspec(3, 2, width_ratios=(1, 2)).subplots()
axes_smo = gs_right[-3:, 1:4].subgridspec(3, 3, width_ratios=(1, 2, 2)).subplots()

axes_int[1, 0].set_ylabel("Intensity threshold", fontsize="small")
for ax, title in zip(axes_int[-1], ("Mask", "Histogram")):
    ax.set_xlabel(title, fontsize="small")

axes_smo[1, 0].set_ylabel("SMO threshold", fontsize="small")
for ax, title in zip(axes_smo[-1], ("Mask", "Histogram", "Normalized\nHistogram")):
    ax.set_xlabel(title, fontsize="small")

# Intensity backgrounds
intensity_thresholds = {
    cmap_int_thres(0.3): scipy.stats.norm.ppf(0.5),
    cmap_int_thres(0.6): scipy.stats.norm.ppf(0.99),
    cmap_int_thres(0.9): 4,
}
intensity_background(
    axes_int, intensity_thresholds, pipeline.intensity, image_true.mask
)

# SMO backgrounds
smo_thresholds = {
    cmap_smo_thres(0.3): smo_rv.ppf(0.1),
    cmap_smo_thres(0.6): smo_rv.ppf(0.9),
    cmap_smo_thres(0.9): 0.75,
}
smo_background(axes_smo, smo_thresholds, pipeline, image_true.mask)

# Panel B intensity thresholds
for color, thres in intensity_thresholds.items():
    for ax, y0, y1 in [(axes_2d.center, -0.1, 0), (axes_2d.top, 1e5, 1e4)]:
        ax.annotate(
            "",
            xy=(thres, y1),
            xytext=(thres, y0),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="simple", color=color),
        )
        ax.axvline(thres, color=color, linestyle="--", linewidth=1, alpha=0.5)

# Panel B SMO thresholds
for color, thres in smo_thresholds.items():
    for ax, x0, x1 in [(axes_2d.center, -5.1, -5), (axes_2d.right, 1e5, 1e4)]:
        ax.annotate(
            "",
            xy=(x1, thres),
            xytext=(x0, thres),
            annotation_clip=False,
            arrowprops=dict(arrowstyle="simple", color=color),
        )
        ax.axhline(thres, color=color, linestyle="--", linewidth=1, alpha=0.5)

# Legend
legend = {
    "Foreground": plt.Line2D([0], [0], color=color_foreground),
    "Background": plt.Line2D([0], [0], color=color_background),
    "All": plt.Rectangle((0, 0), 0, 0, color="gray", alpha=0.5),
    "Intensity threshold": plt.Rectangle(
        (0, 0), 0, 0, color=cmap_int_thres(0.7), alpha=0.5
    ),
    "SMO threshold": plt.Rectangle((0, 0), 0, 0, color=cmap_smo_thres(0.7), alpha=0.5),
}
fig.legend(
    legend.values(),
    legend.keys(),
    fontsize="x-small",
    loc=(axes_2d.center.get_position().x0, axes_smo[-1, 0].get_position().y0),
)

# Panel labels
common.add_subplot_label((*axes_images, *axes_images.cbar_axes), "A")
common.add_subplot_label(axes_2d, "B", offset=(0.0, -0.03))
common.add_subplot_label(axes_int.ravel(), "C", offset=(-0.02, 0))
common.add_subplot_label(axes_smo.ravel(), "D", offset=(-0.02, 0))


common.save_or_show(fig, "figure3")
