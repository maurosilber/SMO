import matplotlib.pyplot as plt
import numpy as np

import smo

from . import common
from .figure4 import data

# Parameters
sigma, window = 1, 7

# Image
pipeline = common.SMOPipeline(data.image, sigma=sigma, window=window)

# SMO distribution
smo_rv = smo.smo_rv((1024, 1024), sigma=sigma, size=window)

# Plot
fig = plt.figure()
left, right = fig.add_gridspec(1, 2, width_ratios=[3, 5], wspace=0.3)

axes = left.subgridspec(2, 2).subplots()
axes[0, 0].imshow(pipeline.intensity, cmap=common.cmap.intensity, rasterized=True)
axes[1, 0].imshow(
    pipeline.angle, cmap=common.cmap.angle, vmin=-np.pi, vmax=np.pi, rasterized=True
)
axes[1, 1].imshow(pipeline.smo, cmap=common.cmap.smo, vmin=0, vmax=1, rasterized=True)

common.gradient_quiver_plot(
    axes[0, 1],
    pipeline.gradient,
    step=2,
    scale=10,
    headwidth=3,
    headlength=2,
    headaxislength=2,
    rasterized=True,
)
common.align_axes(axes[1, 1], axes[0, 1], orientation="x")
common.align_axes(axes[0, 0], axes[0, 1], orientation="y")
axes[0, 1].set(xlim=(265 - 35, 265), ylim=(210, 210 + 35))
axes[1, 0].indicate_inset_zoom(axes[0, 1], edgecolor="k", alpha=1, linewidth=2)

for ax in axes.flat:
    ax.set(xticks=(), yticks=())
    ax.add_patch(
        plt.Rectangle(
            (data.x0, data.y0),
            data.x1 - data.x0,
            data.y1 - data.y0,
            facecolor="none",
            edgecolor="C1",
        )
    )

ax = fig.add_subplot(right)
for name, bg in data.bgs.items():
    ax.plot(*common.ecdf(bg.ravel()), label=name)
ax.set(
    xlim=(193, 220), xlabel="Intensity", ylabel="CDF", title="Background distributions"
)
ax.legend(title="Method")

common.save_or_show(fig, "graphical_abstract")
