import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .. import common
from .data import color_background, color_foreground, image_true, pipeline


def hist2d(axes: common.MarginalAxes, pipeline, bg_mask):
    x = pipeline.intensity.ravel()
    y = pipeline.smo.ravel()
    m = bg_mask.ravel()

    segmentation = [
        ("Foreground", ~m, color_foreground),
        ("Background", m, color_background),
    ]

    xbins, ybins = np.linspace(-5, 15, 100), np.linspace(0, 1, 100)

    # All
    axes.center.hist2d(
        x, y, bins=(xbins, ybins), norm=LogNorm(), cmap="gray", rasterized=True
    )
    for _, mask, color in segmentation:
        Z, X, Y = np.histogram2d(x[mask], y[mask], bins=20)
        axes.center.contour(
            X[:-1], Y[:-1], Z.T, levels=np.geomspace(10, 10000, 5), colors=[color]
        )

    kwargs = dict(histtype="bar", alpha=0.5, color="gray")
    axes.top.hist(x, bins=xbins, **kwargs, label="All")
    axes.right.hist(y, bins=ybins, **kwargs, orientation="horizontal")

    # Separate
    kwargs = dict(histtype="step")
    for label, mask, color in segmentation:
        axes.top.hist(x[mask], bins=xbins, **kwargs, label=label, color=color)
        axes.right.hist(
            y[mask], bins=ybins, **kwargs, orientation="horizontal", color=color
        )

    axes.top.set(yscale="log")
    axes.right.set(xscale="log")

    axes.center.set(
        xlabel="Intensity",
        ylabel="SMO value",
        xlim=(-5, 15),
        xticks=np.linspace(-5, 15, 5),
        xticklabels=(-5, "", "", "", 15),
        ylim=(0, 1),
        yticks=np.linspace(0, 1, 5),
        yticklabels=(0, *3 * ("",), 1),
    )
    axes.right.set(
        xlim=(1, 1e4),
        xticks=10 ** np.arange(5),
        xticklabels=(),
        ylim=(0, 1),
        yticks=np.linspace(0, 1, 5),
        yticklabels="",
    )
    axes.top.set(
        xlim=(-5, 15),
        xticks=(np.linspace(-5, 15, 5)),
        xticklabels=(),
        ylim=(1, 1e4),
        yticks=10 ** np.arange(5),
        yticklabels=(),
    )

    common.set_logticks(axes.right.xaxis)
    common.set_logticks(axes.top.yaxis)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(3, 2))
    axes = common.MarginalAxes(ax)
    hist2d(axes, pipeline, image_true.mask)
    plt.show()
