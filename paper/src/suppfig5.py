import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import ImageGrid

from . import common
from .figure2.data import pipelines

fig = plt.figure()
axes = ImageGrid(
    fig, 111, (1, 4), share_all=True, cbar_mode="single", axes_pad=0.2, cbar_pad=0.1
)

norm = TwoSlopeNorm(1, 0.8, 1.2)
cmap = "seismic"
for ax, (name, pipeline) in zip(axes, pipelines.items()):
    ax.set(title=name, xlabel="CDF(Intensity)")

    image_size = pipeline.intensity.size
    intensity = np.argsort(pipeline.intensity.ravel()) / image_size
    smo = np.argsort(pipeline.smo.ravel()) / image_size
    ax.hist2d(
        intensity,
        smo,
        bins=np.linspace(0, 1, 100),
        density=True,
        norm=norm,
        cmap=cmap,
        rasterized=True,
    )

axes[0].set(ylabel="CDF(SMO)", yticks=(0, 0.5, 1))
axes[0].cax.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    label="Probability Density",
    ticks=(0.8, 0.9, 1, 1.1, 1.2),
)

common.save_or_show(fig, "suppfig5")
