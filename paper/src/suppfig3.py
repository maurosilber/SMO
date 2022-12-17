import matplotlib.pyplot as plt
import numpy as np

import smo

from . import common

# Image generation
n = 128
profile = np.zeros(2 * n)
profile[n:] = 0.1 * np.arange(0, n, dtype=float)
image = np.tile(profile, (profile.size, 1))
rng = np.random.default_rng(seed=42)
image = rng.normal(image)


sizes = (0, 0.5, 0.8, 1, 10)

cmap = plt.get_cmap("viridis")
sizes = {s: cmap(c) for s, c in zip(sizes, np.linspace(0, 1, len(sizes)))}

fig = plt.figure()
fig.set_figheight(2)

A, B = fig.add_gridspec(1, 2, width_ratios=(1, 6), wspace=0.5)

axes_image, axes_profile = A.subgridspec(2, 1, hspace=0.5).subplots()
axes_image.imshow(image)
axes_profile.plot(image[0], linewidth=0.5)
axes_profile.plot(profile)


top, bottom = B.subgridspec(2, 1, height_ratios=(1, 1.5))
axes_images = top.subgridspec(1, 5).subplots()
left, right = bottom.subgridspec(1, 2, width_ratios=(2, 1), wspace=0.5)
axes_pdf = left.subgridspec(1, 2, wspace=0.4).subplots()
axes_cdf = fig.add_subplot(right)

axes_hists = {
    axes_cdf: {"cumulative": True},
    axes_pdf[0]: {"cumulative": False},
    axes_pdf[1]: {"cumulative": False},
}

axes_pdf[0].set(ylabel="PDF")
axes_pdf[1].set(yscale="log")
axes_cdf.set(ylabel="CDF")

for ax in axes_hists:
    ax.set(xlabel="Intensity [a.u.]")

bins = np.concatenate((np.linspace(-5, 5, 100, endpoint=False), np.linspace(5, 15, 20)))

for ax, hist_kw in axes_hists.items():
    ax.hist(
        image[:, :n].flat,
        bins=bins,
        density=True,
        color="gray",
        alpha=0.5,
        label="Ground\ntruth",
        **hist_kw,
    )

for ax, (size, color) in zip(axes_images, sizes.items()):
    smo_image = smo.smo(image, sigma=size, size=7)
    smo_rv = smo.smo_rv((1024, 1024), sigma=size, size=7)
    bg_dist = image[smo_image < smo_rv.ppf(0.05)]

    ax.imshow(smo_rv.cdf(smo_image), cmap=common.cmap.smo, rasterized=True)

    for ax_h, hist_kw in axes_hists.items():
        ax_h.hist(
            bg_dist,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=0.5,
            color=color,
            **hist_kw,
        )

    ax.set(xticks=(), yticks=(), title=f"$\\sigma = {size}$")
    for spine in ax.spines.values():
        spine.set(color=color, linewidth=3)

axes_cdf.legend(loc="lower right", fontsize="small")

common.add_subplot_label((axes_image, axes_profile, *axes_images), "A")
common.add_subplot_label((*axes_images, *axes_pdf, axes_cdf), "B")

plt.setp(axes_pdf[1].yaxis.get_ticklabels(), fontsize="x-small")

common.save_or_show(fig, "suppfig3")
