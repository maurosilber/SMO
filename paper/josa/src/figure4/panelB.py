import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .. import common
from .data import otsu_threshold, pipeline, smo_rv


def hist2d(axes: common.MarginalAxes, pipeline, smo_rv):
    x = pipeline.intensity.ravel()
    y = pipeline.smo.ravel()

    # All
    xcounts, xbins, _ = axes.top.hist(x, bins=100, color="gray")
    ycounts, ybins, _ = axes.right.hist(
        y, bins=np.linspace(0, 1, 100), color="gray", orientation="horizontal"
    )
    axes.center.hist2d(
        x, y, bins=(xbins, ybins), norm=LogNorm(), cmap="gray", rasterized=True
    )

    # SMO random
    smo_thres = smo_rv.ppf(0.05)
    axes.center.axhline(smo_thres, color="C0")
    axes.center.fill_between(
        [0, 1100], -1, smo_thres, zorder=100, color="C0", alpha=0.5
    )

    h, b = smo_rv._histogram
    b = b[:-1]
    h = h / h.max() * ycounts.max()
    axes.right.plot(h, b, linestyle="--", color="C0")
    ix = np.searchsorted(b, smo_thres)
    axes.right.hlines(smo_thres, 0, h[ix], color="C0")
    axes.right.fill_betweenx(b[:ix], h[:ix], zorder=100, color="C0", alpha=0.5)

    # Otsu threshold
    for ax in (axes.center, axes.top):
        ax.axvline(otsu_threshold, color="C2")

    axes.center.set(xlabel="Intensity", ylabel="SMO values")
    axes.center.tick_params(labelsize="small")
    axes.top.set(yscale="log")

    for ax in (axes.center, axes.top):
        ax.set(xlim=(0, 1030))

    for ax in (axes.center, axes.right):
        ax.set(ylim=(-0.02, 1.02))

    for ax in (axes.top, axes.right):
        ax.set(xticks=(), yticks=())

    common.set_logticks(axes.top.yaxis)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(2, 2))
    axes = common.MarginalAxes(ax)
    hist2d(axes, pipeline, smo_rv)
    fig.show()
