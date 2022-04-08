import matplotlib.pyplot as plt
import numpy as np

import smo

from .. import common

# Colors
color_foreground = "C3"
color_background = "C2"
cmap_int_thres = plt.get_cmap("Oranges")
cmap_smo_thres = plt.get_cmap("Blues")

rng = np.random.default_rng(seed=42)

# Parameters
sigma, window = 1, 7

# Image
image_true = common.generate_image(amplitude=10, pad=12)
image = rng.normal(image_true.data)
pipeline = common.SMOPipeline(image, sigma=sigma, window=window)

# SMO distribution
smo_rv = smo.smo_rv((1024, 1024), sigma=sigma, size=window)

if __name__ == "__main__":
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(5, 1))

    line = len(image) // 2

    axes[0].set(title="Ground truth")
    axes[0].imshow(image_true.data)
    axes[0].axhline(line, ls="--")

    axes[1].set(title="With additive\nnormal noise")
    axes[1].imshow(image)
    axes[1].axhline(line, color="C1", ls="--")

    axes[2].set(title="Line profiles")
    axes[2].plot(image[line])
    axes[2].plot(image_true.data[line])

    plt.show()
