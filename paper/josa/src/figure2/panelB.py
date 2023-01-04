import matplotlib.pyplot as plt
import numpy as np

from .. import common


def panelB(axes: common.MarginalAxes, ax_bottom, pipeline):
    x, y = pipeline.intensity.ravel(), pipeline.smo.ravel()
    bins = np.linspace(0, 1, 200), np.linspace(0, 1, 200)

    quantiles = {
        0.1: {
            "color": "C0",
            "s": 30,
            "marker": "o",
            "edgecolor": "C0",
            "facecolor": "none",
        },
        0.5: {"color": "C1", "s": 30, "marker": "+"},
        0.9: {"color": "C2", "s": 30, "marker": "x"},
    }

    _, _, yhist = hist2d(axes, x, y, bins)
    for q, kwargs in quantiles.items():
        conditional_dists(
            axes, ax_bottom, x=x, y=y, bins=bins, q=q, width=0.1, yhist=yhist, **kwargs
        )

    kwargs = {
        "xlim": (0, 1),
        "ylim": (0, 0.45),
        "xticks": (0, 1),
        "yticks": (0, 0.2, 0.4),
    }

    axes.center.set(**kwargs)
    axes.top.set(xlim=kwargs["xlim"], ylabel="PDF")
    axes.right.set(ylim=kwargs["ylim"], xlabel="PDF")
    ax_bottom.set(ylim=(0, 1.5))
    ax_bottom.legend(
        title="SMO\nquantile",
        loc=(1.05, 0),
        fontsize="x-small",
        title_fontsize="x-small",
    )


def hist2d(axes: common.MarginalAxes, x, y, bins):
    color = "gray"

    hist2d, *_ = axes.center.hist2d(x, y, bins=bins, cmap="cividis", rasterized=True)
    xhist, *_ = axes.top.hist(x, bins=bins[0], color=color, rasterized=True)
    yhist, *_ = axes.right.hist(
        y, bins=bins[1], color=color, orientation="horizontal", rasterized=True
    )

    axes.center.set(xlabel="Intensity value", ylabel="SMO value")
    return hist2d, xhist, yhist


def conditional_dists(
    axes: common.MarginalAxes, ax, *, x, y, bins, q, width, color, yhist, **kwargs
):
    xbins, ybins = bins

    a, m, b = np.quantile(y, (q - width / 2, q, q + width / 2))
    mask = (a <= y) & (y < b)
    ax.hist(
        x[mask],
        bins=xbins,
        density=True,
        histtype="step",
        label=f"{q:.1f}",
        linewidth=0.5,
    )
    ax.set(xlabel="Intensity value", ylabel="PDF", yticks=(0, 1.0), ylim=(0, 1.05))

    axes.center.annotate(
        "",
        xy=(0, m),
        xytext=(-0.2, m),
        annotation_clip=False,
        arrowprops=dict(arrowstyle="simple", color=color),
    )

    alpha = 0.5
    axes.center.fill_between(xbins, a, b, color=color, alpha=alpha, linewidth=0)
    t = np.linspace(a, b, 50)
    axes.right.fill_betweenx(
        t,
        np.interp(t, (ybins[1:] + ybins[:-1]) / 2, yhist),
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=100,
    )


if __name__ == "__main__":
    from .data import pipeline

    fig = plt.figure(figsize=(1, 2.5))
    gs = fig.add_gridspec(2, 1, hspace=0.4)
    axes_2d = common.MarginalAxes(fig.add_subplot(gs[0]))

    plt.draw()
    pos = axes_2d.center.get_position()
    pos_b = gs[1].get_position(fig)
    ax_bottom = fig.add_axes([pos_b.x0, pos_b.y0, pos.width, pos.height])

    panelB(axes_2d, ax_bottom, pipeline)
    plt.show()
