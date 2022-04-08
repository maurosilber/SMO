import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

from smo.smo import _normalized_gradient


def save_or_show(figure, filename):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.save:
        path = Path("figures") / filename
        path.mkdir(parents=True, exist_ok=True)

        for ext in ("png", "svg", "pdf"):
            figure.savefig(path / f"{filename}.{ext}")
    else:
        plt.show()


# Colormaps
class cmap:
    intensity = "plasma"
    angle = "twilight"
    smo = "cividis"


def generate_image(*, amplitude, background=0, pad=12, width=0.5):
    # Coordinates
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2

    mask = R2 > x[-1] ** 2
    image = np.where(mask, background, amplitude * np.exp(-R2 / width))
    if pad > 0:
        image = np.pad(image, pad, constant_values=background)
        mask = np.pad(mask, pad, constant_values=True)
    return np.ma.masked_array(image, mask)


class SMOPipeline:
    def __init__(self, image, *, sigma, window):
        self.intensity = image
        self.sigma = sigma
        self.window = window

    @property
    def smoothed_image(self):
        return ndimage.gaussian_filter(self.intensity.astype(float), self.sigma)

    @property
    def gradient(self):
        return np.array(_normalized_gradient(self.smoothed_image))

    @property
    def angle(self):
        return np.arctan2(*self.gradient)

    @property
    def averaged_gradient(self):
        size = (1,) + self.intensity.ndim * (self.window,)
        return ndimage.uniform_filter(self.gradient, size)

    @property
    def averaged_angle(self):
        return np.arctan2(*self.averaged_gradient)

    @property
    def smo(self):
        return np.linalg.norm(self.averaged_gradient, axis=0)


def gradient_quiver_plot(ax, gradient_image, *, step: int, **quiver_kwargs):
    quiver_kwargs = {
        "scale": 10,
        "width": 0.02,
        "headwidth": 3,
        "headlength": 3,
        "headaxislength": 3,
        **quiver_kwargs,
    }

    n, m = gradient_image.shape[1:]
    image = gradient_image[:, ::step, ::step]
    angle = np.arctan2(*image)

    return ax.quiver(
        *np.mgrid[:n:step, :m:step][::-1],
        *image[::-1],
        angle,
        cmap=cmap.angle,
        norm=plt.Normalize(-np.pi, np.pi),
        scale_units="width",
        pivot="mid",
        **quiver_kwargs,
    )


class MarginalAxes:
    def __init__(self, ax, *, top="40%", right="40%"):
        divider = make_axes_locatable(ax)
        self.center = ax
        self.top = divider.append_axes("top", top, pad=0)
        self.right = divider.append_axes("right", right, pad=0)

        for ax in (self.top, self.right):
            ax.set(xticks=(), yticks=())

    def __iter__(self):
        yield self.center
        yield self.top
        yield self.right


def add_subplot_label(axes, label: str, *, offset=(0, 0), text_prop={}):
    text_prop = {
        "verticalalignment": "bottom",
        "horizontalalignment": "right",
        "fontsize": "xx-large",
        "fontweight": "bold",
        **text_prop,
    }

    if isinstance(axes, plt.Axes):
        ax, extra_axes = axes, ()
    else:
        ax, *extra_axes = axes

    fig = ax.figure
    bbox = ax.get_tightbbox(fig.canvas.get_renderer(), bbox_extra_artists=extra_axes)
    bbox = bbox.transformed(fig.transFigure.inverted())
    dx, dy = offset
    return fig.text(bbox.x0 + dx, bbox.y1 + dy, label, **text_prop)


def set_logticks(axis, *, subs=np.linspace(0, 1, 10), numticks=12):
    locator = matplotlib.ticker.LogLocator(subs=subs, numticks=numticks)
    axis.set_minor_locator(locator)
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def align_axes(ax_source: plt.Axes, ax_target: plt.Axes, *, orientation: str):
    pos0 = ax_source.get_position()
    pos1 = ax_target.get_position()

    if orientation == "x":
        pos1.x0 = pos0.x0
        pos1.x1 = pos0.x1
    elif orientation == "y":
        pos1.y0 = pos0.y0
        pos1.y1 = pos0.y1
    else:
        raise ValueError(f'Orientation {orientation} not valid. Use: "x" or "y".')

    ax_target.set_position(pos1)


def equal_hist(y, *, nbins):
    if y[0] == 0:
        ix = np.argmax(y > 0)
        y = y[ix:]
    else:
        ix = 0

    prob = np.empty(y.size + 1)
    prob[0] = 0
    prob[1:] = np.cumsum(y)
    prob /= prob[-1]

    cuts = np.linspace(0, 1, nbins + 1, endpoint=True)
    bins = np.searchsorted(prob, cuts, side="left")
    bins = np.unique(bins)
    density = np.diff(prob[bins]) / np.diff(bins)
    return density, bins + ix


def plot_equal_hist(ax, y, nbins, **step):
    density, bins = equal_hist(y, nbins=nbins)
    ax.step(np.append(bins, bins[-1]), np.pad(density, 1), **step)


def ecdf(x):
    return np.sort(x), np.linspace(0, 1, x.size)
