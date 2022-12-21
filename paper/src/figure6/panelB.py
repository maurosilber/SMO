import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.io import imread

from .. import common
from ..BBBC025 import channel_order, image_directory, method_order
from ..common import ecdf
from .data import df, df_histograms, df_low


def plot_image(ax, file):
    image = imread(file)
    ax.imshow(image, cmap=common.cmap.intensity, norm=LogNorm(), rasterized=True)
    ax.set(xticks=(), yticks=())


def plot_histogram(ax, channel, df_channel, *, nbins):
    dir = next(image_directory.rglob(f"*_{channel}"))
    for _, file in zip(range(50), dir.glob("*.tif")):
        image = imread(file)
        y = np.bincount(image.flat)
        common.plot_equal_hist(
            ax, y, nbins=nbins, color="k", alpha=0.1, rasterized=True
        )

    ax.set(xscale="log", yscale="log", xlim=(3e2, 8e4), yticks=(1e-2, 1e-4, 1e-6, 1e-8))
    ax.set_xlabel("Intensity", fontsize="x-small")


def plot_cdf(ax, df_channel):
    dg = df_channel[method_order]
    for i, (name, col) in enumerate(dg.items()):
        ax.plot(*ecdf(col), label=name, rasterized=False, zorder=10 - i)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", linestyle="--")


def panelB(fig, sps, **gridspec_kw):
    with plt.rc_context({"ytick.labelsize": "x-small", "xtick.labelsize": "xx-small"}):
        gs = sps.subgridspec(4, 1, height_ratios=[2, 1, 0.7, 2.5], **gridspec_kw)
        axes_images = gs[0].subgridspec(1, 5, **gridspec_kw).subplots(sharey=True)
        axes_hists = gs[1].subgridspec(1, 5, **gridspec_kw).subplots(sharey=True)
        axes_cdfs = gs[3].subgridspec(1, 5, **gridspec_kw).subplots(sharey=True)

        for channel, df_channel in df_low.groupby("channel"):
            ax = axes_images[channel_order.index(channel)]
            plot_image(ax, df_channel.iloc[0].file)
            ax.set(title=channel)

        for ax_base, *rest in zip(axes_images, axes_hists, axes_cdfs):
            for ax in rest:
                common.align_axes(ax_base, ax, orientation="x")

        for channel, df_channel in df_histograms.groupby("channel"):
            ax = axes_hists[channel_order.index(channel)]
            plot_histogram(ax, channel, df_channel, nbins=1000)
            ax.set_xlabel("Intensity", fontsize="x-small")

        for channel, df_channel in df.groupby("channel"):
            ax = axes_cdfs[channel_order.index(channel)]
            plot_cdf(ax, df_channel)
            xlim, xticks = xlims[channel]
            ax.set(xlim=xlim, xticks=xticks)
            ax.set_xlabel("Median background", fontsize="x-small")

        axes_hists[0].set(ylabel="PDF")
        axes_cdfs[0].set(ylabel="CDF")

        axes_cdfs[-1].legend(
            # title="Methods",
            fontsize="x-small",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )

    return np.array((axes_images, axes_hists, axes_cdfs))


xlims = {
    "ERSyto": ((800, 2200), [1000, 1500, 2000]),
    "ERSytoBleed": ((900, 3100), [1000, 2000, 3000]),
    "Mito": ((1100, 1900), [1250, 1500, 1750]),
    "PhGolgi": ((2400, 7600), [2500, 5000, 7500]),
    "Hoechst": ((495, 535), [500, 515, 530]),
}


if __name__ == "__main__":
    fig = plt.figure()
    fig.set_figheight(3)
    sps = fig.add_gridspec()[0]
    panelB(fig, sps)
    fig.show()
