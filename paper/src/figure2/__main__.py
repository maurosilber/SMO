import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import common
from .data import pipeline, pipelines
from .panelA import image_pipeline
from .panelB import panelB
from .panelC import dist_histograms

fig = plt.figure()
fig.set_figheight(3)
gs = fig.add_gridspec(1, 3, wspace=0.5)

# Image pipeline
axes_images = ImageGrid(
    fig,
    gs[0].get_position(fig).bounds,
    (3, 1),
    axes_pad=0.1,
    cbar_mode="each",
    cbar_location="left",
)
image_pipeline(axes_images, pipeline, slice=(slice(0, 64), slice(0, 64)))

# 2D Histogram
with plt.rc_context({"ytick.labelsize": "small", "xtick.labelsize": "small"}):
    gs_top, gs_bottom = gs[0, 1].subgridspec(2, 1)
    axes_2d = common.MarginalAxes(fig.add_subplot(gs_top))
    ax_bottom = fig.add_axes(gs_bottom.get_position(fig).shrunk(1 / 1.4, 1 / 1.4))
    panelB(axes_2d, ax_bottom, pipeline)

# # 1D distributions
axes_dist1d = gs[0, 2].subgridspec(2, 1, hspace=0.5).subplots()
dist_histograms(axes_dist1d, pipelines)
axes_dist1d[1].legend(fontsize="small", handlelength=1)

common.add_subplot_label((*axes_images, *axes_images.cbar_axes), "A")
common.add_subplot_label((*axes_2d, ax_bottom), "B")
common.add_subplot_label(axes_dist1d, "C")

common.save_or_show(fig, "figure2")
