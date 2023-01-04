import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import common
from . import data
from .panelA import image_pipeline
from .panelB import hist2d
from .panelC import masks
from .panelD import background_estimations

fig = plt.figure()
fig.set_figheight(3)
gs = fig.add_gridspec(1, 3, width_ratios=[2, 4, 3], wspace=0.6)

# Image pipeline
axes_images = ImageGrid(
    fig,
    gs[0].get_position(fig).bounds,
    (3, 1),
    axes_pad=0.05,
    cbar_mode="each",
    cbar_location="left",
)
image_pipeline(axes_images, data.pipeline)

# Manual selection
for ax in axes_images:
    ax.add_patch(
        plt.Rectangle(
            (data.x0, data.y0),
            data.x1 - data.x0,
            data.y1 - data.y0,
            facecolor="none",
            edgecolor="C1",
        )
    )


top, bottom = gs[1].subgridspec(2, 1, hspace=1, height_ratios=(2, 1))

# 2D Histogram
axes_2d = common.MarginalAxes(fig.add_subplot(top))
hist2d(axes_2d, data.pipeline, data.smo_rv)

# Masks
ax, *extra_axes = axes_2d
bbox = ax.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=extra_axes)
bbox = bbox.transformed(fig.transFigure.inverted())
y0, _, y1, _ = bbox.bounds
_, x0, _, x1 = bottom.get_position(fig).bounds

axes_masks = ImageGrid(
    fig,
    (y0, x0, y1, x1),
    (1, 3),
    axes_pad=0.05,
)
masks(
    axes_masks,
    data.pipeline.intensity,
    data.otsu_mask,
    data.smo_mask,
    data.rolling_ball,
)


# Background distribution
axes_hists = (
    gs[2].subgridspec(12, 1)[1:-1].subgridspec(2, 1, hspace=0.1).subplots(sharex=True)
)
background_estimations(axes_hists, data.bgs)

for label, axes in {
    "A": (*axes_images, *axes_images.cbar_axes),
    "B": axes_2d,
    "C": axes_masks,
    "D": axes_hists,
}.items():
    common.add_subplot_label(axes, label)

common.save_or_show(fig, "figure4")
