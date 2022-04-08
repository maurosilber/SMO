import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.io import imread

from . import common
from .BBBC025 import channel_order
from .figure6.data import df_extreme, df_low


def plot_channel(dg, axes):
    for ax, file in zip(axes, dg.file):
        image = imread(file)
        ax.imshow(image, norm=LogNorm(), cmap=common.cmap.intensity, rasterized=True)
        ax.set(xticks=(), yticks=())


fig = plt.figure()
fig.set_figheight(4)

axes_low = ImageGrid(fig, 211, (5, 10))
axes_extreme = ImageGrid(fig, 212, (5, 10))

for axes, df in [(axes_low, df_low), (axes_extreme, df_extreme)]:
    for channel, dg in df.groupby("channel"):
        row_ix = channel_order.index(channel)
        plot_channel(dg, axes=axes.axes_row[row_ix])
        axes.axes_row[row_ix][0].set_ylabel(
            channel, rotation=0, ha="right", va="center"
        )

fig.suptitle("Low density of cells", y=axes_low[0].get_position().y1 + 0.05)
fig.supxlabel("High density of cells", y=axes_extreme[-1].get_position().y0 - 0.05)

common.save_or_show(fig, "suppfig8")
