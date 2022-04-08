import matplotlib.pyplot as plt
import numpy as np

from .. import common


def step_hist(ax, x, bins, **kwargs):
    counts, bins = np.histogram(x, bins, density=True)
    ax.step(bins[:-1], counts, **kwargs)


def dist_histograms(axes, pipelines: common.SMOPipeline):
    for name, pipeline in pipelines.items():
        step_hist(
            axes[0], pipeline.intensity.flat, bins=np.linspace(-3, 6, 300), label=name
        )
        step_hist(axes[1], pipeline.smo.flat, bins=np.linspace(0, 1, 100), label=name)

    axes[0].set(xlabel="Intensity", xticks=(-3, 0, 3, 6), yticks=())
    axes[1].set(xlabel="SMO", yticks=())


if __name__ == "__main__":
    from .data import pipelines

    fig, axes = plt.subplots(2, 1, figsize=(1, 2), gridspec_kw={"hspace": 0.5})
    dist_histograms(axes, pipelines)
    plt.show()
