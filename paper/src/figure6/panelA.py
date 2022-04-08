import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.io import imread

import smo

from .. import common
from ..BBBC025 import image_directory
from .data import df


class StackedAxes:
    def __init__(self, number, ratio, bbox, *, fig=None, transparent=False):
        if fig is None:
            fig = plt.gcf()

        if len(ratio) == 1:
            ratio = (ratio, ratio)
        elif len(ratio) != 2:
            raise ValueError

        self.fig = fig
        self.bbox = bbox
        self.axes = list(
            self._yield_axes(fig, number, ratio, bbox, transparent=transparent)
        )

    def __iter__(self):
        yield from self.axes

    @staticmethod
    def _yield_axes(fig, number, ratio, bbox, *, transparent=False):
        rx, ry = ratio
        ox = bbox.x0 + bbox.width * np.linspace(0, 1 - rx, number)
        oy = bbox.y0 + bbox.height * np.linspace(0, 1 - ry, number)
        for i, o in enumerate(zip(ox, oy)):
            ax = fig.add_axes([*o, bbox.width * rx, bbox.height * ry], zorder=-i)
            if transparent:
                ax.set_facecolor("none")
                ax.spines.top.set_visible(False)
                ax.spines.right.set_visible(False)
            yield ax

    def set_title(self, title, pad=None, **kwargs):
        rcParams = plt.rcParams
        fontdict = {
            "fontsize": rcParams["axes.titlesize"],
            "fontweight": rcParams["axes.titleweight"],
            "verticalalignment": "baseline",
            "horizontalalignment": rcParams["axes.titlelocation"],
        }

        if pad is None:
            pad = rcParams["axes.titlepad"]
            _, pad = self.fig.transFigure.inverted().transform((0, pad))

        x = self.bbox.x0 + self.bbox.width / 2
        y = self.bbox.y1 + pad
        self.fig.text(x, y, title, fontdict=fontdict, **kwargs)


def plot_image(ax, image):
    ax.imshow(image, norm=LogNorm(), cmap=common.cmap.intensity, rasterized=True)
    ax.set(xticks=(), yticks=())


def plot_histogram(ax, image, bins):
    pipeline = common.SMOPipeline(image, sigma=0, window=7)
    smo_rv = smo.smo_rv((1024, 1024), sigma=0, size=7)
    bg = pipeline.intensity[pipeline.smo < smo_rv.ppf(0.05)]

    counts, bins, _ = ax.hist(bg, bins=bins)
    median = np.median(bg)
    ix = np.searchsorted(bins, median)
    ax.vlines(median, 0, counts[ix], color="C1")
    ax.set(xticks=(), yticks=())


def panelA(fig, sps):
    channel = "ERSyto"

    dir = next(image_directory.rglob(f"*_{channel}"))
    files = dir.glob("*.tif")

    smo_medians = df[df.channel == channel].SMO

    gs = sps.subgridspec(1, 4, width_ratios=[1, 1, 0.1, 1])
    n, ratio = 6, (0.3, 0.5)

    axes_images = StackedAxes(n, ratio, gs[0].get_position(fig), fig=fig)
    axes_images.set_title("Intensity\nimages", pad=0.02)
    axes_bg_hist = StackedAxes(
        n, ratio, gs[1].get_position(fig), fig=fig, transparent=True
    )
    axes_bg_hist.set_title("Background\ndistribution", pad=0.02)
    for ax_im, ax_bg, file in zip(axes_images, axes_bg_hist, files):
        image = imread(file)
        plot_image(ax_im, image)
        plot_histogram(ax_bg, image, bins=np.arange(600, 1000, 10))

    # Histogram of medians
    ax_hist = fig.add_axes(gs[-1].get_position(fig))
    bins = np.arange(*smo_medians.quantile((0, 0.95)), 10)
    ax_hist.hist(smo_medians, bins=bins, color="C1")
    ax_hist.set(title="Median\nbackground", xticks=(), yticks=())

    return (*axes_images, *axes_bg_hist, ax_hist)


if __name__ == "__main__":
    fig = plt.figure()
    fig.set_figheight(1)
    sps = fig.add_gridspec()[0]
    panelA(fig, sps)
    fig.show()
