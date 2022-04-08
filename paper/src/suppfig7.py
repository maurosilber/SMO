import matplotlib.pyplot as plt

from . import common
from .BBBC025 import channel_order, load_summary, method_order

df = load_summary()

fig, axes = plt.subplots(
    len(method_order),
    len(channel_order),
    sharex=True,
    sharey="col",
    gridspec_kw=dict(hspace=0.1, wspace=0.4),
)
fig.set_figheight(5)

smo_breaking_point = {
    "ERSyto": 0.10,
    "ERSytoBleed": 0.12,
    "Hoechst": 0.15,
    "Mito": 0.03,
    "PhGolgi": 0.12,
}

for channel, dg in df.groupby("channel"):
    axes_col = axes[:, channel_order.index(channel)]
    axes_col[0].set(title=channel)

    for ax, method in zip(axes_col, method_order):
        ax.scatter(dg.segmentation_size, dg[method], s=1, rasterized=True)
        ax.axvline(smo_breaking_point[channel], linestyle="--", color="k")
        ax.tick_params(labelsize="xx-small", direction="in", pad=2)

for ax, method in zip(axes[:, 0], method_order):
    ax.set(ylabel=method)

fig.supxlabel("Background area fraction")

common.save_or_show(fig, "suppfig7")
