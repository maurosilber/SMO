import matplotlib.pyplot as plt
import numpy as np


def background_estimations(axes, bgs):
    for ax, scale, cumulative in zip(axes, ("linear", "log"), (True, False)):
        ax.set(yscale=scale, xlim=(193, 240))
        ax.tick_params(labelsize="small")

        for title, bg in bgs.items():
            ax.hist(
                bg.flat,
                bins=np.arange(190, 260),
                histtype="step",
                density=True,
                label=title,
                cumulative=cumulative,
            )

    axes[0].set_title("Background distributions", fontsize="small")
    axes[0].set(ylabel="CDF")
    axes[1].set(ylabel="PDF", xlabel="Intensity")

    axes[0].legend(loc="lower right", fontsize="x-small", handlelength=1)


if __name__ == "__main__":
    from .data import bgs

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(1, 2))
    background_estimations(axes, bgs)
    axes[0].legend(bbox_to_anchor=(1, 1))
    fig.show()
