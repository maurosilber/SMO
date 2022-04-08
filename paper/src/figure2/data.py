import numpy as np

from .. import common

size = (2048, 2048)  # Image size
sigma, window = 0, 7  # SMO parameters

# Image generation
rng = np.random.default_rng(seed=42)
distributions = (rng.uniform, rng.normal, rng.exponential, rng.lognormal)
images: dict[str, np.ndarray] = {f.__name__: f(size=size) for f in distributions}
pipelines = {
    name: common.SMOPipeline(image, sigma=sigma, window=window)
    for name, image in images.items()
}

# Shown image
pipeline = pipelines["uniform"]
