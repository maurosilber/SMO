import matplotlib.pyplot as plt
import numpy as np

from . import common

# True image
n = 32
n2 = n // 2
true_image = np.zeros((n, n))
true_image[:, n2:] = np.linspace(0, 20, n2)

# Pipeline
rng = np.random.default_rng(42)
noisy_image = rng.normal(true_image)
pipeline = common.SMOPipeline(noisy_image, sigma=0, window=5)

fig, (ax2d, ax1d) = plt.subplots(
    2, 4, sharex=True, gridspec_kw={"height_ratios": (3, 1), "wspace": 0.5}
)

# Line plots
ax1d[0].plot(pipeline.intensity[n2])
ax1d[0].plot(true_image[n2])
ax1d[3].plot(pipeline.smo[n2])

for ax in ax1d:
    ax.set(xticks=(0, 16, 32))

ax1d[1].remove()
ax1d[2].remove()
ax1d[3].set(yticks=(0, 1))
ax1d[3].yaxis.set_ticks_position("right")

# Image plots
for ax, title in zip(
    ax2d, ("Intensity image", "Gradient direction", "Averaged direction", "SMO image")
):
    ax.set(xticks=(), yticks=())
    ax.set_title(title, fontsize="small")

norm = plt.Normalize(pipeline.intensity.min(), pipeline.intensity.max())

# Noisy image
map_noisy = ax2d[0].imshow(
    pipeline.intensity, cmap=common.cmap.intensity, norm=norm, rasterized=True
)

cax = fig.add_axes(ax2d[0].get_position().translated(-0.02, 0).shrunk(0.05, 1))
plt.colorbar(map_noisy, cax=cax, orientation="vertical")
cax.yaxis.set_ticks_position("left")
cax.yaxis.set_label_position("left")

# Gradient image
map_gradient = common.gradient_quiver_plot(ax2d[1], pipeline.gradient, step=2)

ax2d[1].set(aspect="equal")
cax2 = fig.add_axes(ax2d[1].get_position().translated(0, -0.08).shrunk(1, 0.05))
plt.colorbar(
    map_gradient,
    cax=cax2,
    orientation="horizontal",
    ticks=(-np.pi, 0, np.pi),
    label="Angle",
)
cax2.set_xlim(-np.pi, np.pi)
cax2.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])


# Averaged gradient image
map_gradient = common.gradient_quiver_plot(ax2d[2], pipeline.averaged_gradient, step=2)
ax2d[2].set(aspect="equal")
cax2 = fig.add_axes(ax2d[2].get_position().translated(0, -0.08).shrunk(1, 0.05))
plt.colorbar(
    map_gradient,
    cax=cax2,
    orientation="horizontal",
    ticks=(-np.pi, 0, np.pi),
    label="Angle",
)
cax2.set_xlim(-np.pi, np.pi)
cax2.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])

# SMO image
map_smo = ax2d[3].imshow(
    pipeline.smo, cmap=common.cmap.smo, vmin=0, vmax=1, rasterized=True
)

# Averaging regions
w = 2

for ax in ax2d[1:3]:
    rect = plt.Rectangle(
        (5, 5), width=3 * w, height=3 * w, facecolor="none", edgecolor=(1.0, 0, 0)
    )
    ax.add_patch(rect)
    for x, y in np.ndindex((3, 3)):
        rect = plt.Rectangle(
            (5 + w * x, 5 + w * y),
            width=w,
            height=w,
            facecolor="none",
            edgecolor=(1.0, 0, 0, 0.5),
        )
        ax.add_patch(rect)

for x, y in np.ndindex((3, 3)):
    rect = plt.Rectangle(
        (3 + w * (x + 1), 3 + w * (y + 1)),
        width=w,
        height=w,
        facecolor=(1.0, 0, 0, 0.5),
        edgecolor="none",
    )
    ax2d[1].add_patch(rect)


for x, y in np.ndindex((1, 1)):
    rect = plt.Rectangle(
        (3 + w * (x + 2), 3 + w * (y + 2)),
        width=w,
        height=w,
        facecolor=(1.0, 0, 0, 0.5),
        edgecolor="none",
    )
    ax2d[2].add_patch(rect)


pos = ax2d[3].get_position()
cax3 = fig.add_axes(pos.translated(pos.width + 0.01, 0).shrunk(0.05, 1))
plt.colorbar(map_smo, cax=cax3, orientation="vertical", ticks=(0, 1))
cax3.yaxis.set_ticks_position("right")
cax3.yaxis.set_label_position("right")
cax3.set_ylim(0, 1)

for label, ax in zip("ABCD", ax2d):
    common.add_subplot_label(ax, label, offset=(-0.01, 0))

common.save_or_show(fig, "figure1")
