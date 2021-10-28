# SMO: robust and unbiased background distribution estimation by exploiting lack of local correlation

SMO is a Python package that implements the Silver Mountain Operator (SMO), which allows to recover an unbiased estimation of the background intensity distribution in a robust way.

## Installation

It can be installed with `pip` from PyPI:

```
pip install smo
```

## Plugins
### Napari

A [napari](https://napari.org) plugin is available. Just `pip` install this package.

## CellProfiler

A [CellProfiler](https://cellprofiler.org) plugin in available in the `smo/plugins/cellprofiler` folder.

Save [this file](https://github.com/maurosilber/SMO/tree/main/smo/plugins/cellprofiler/smo.py) into your CellProfiler plugins folder.

## Example

```python
import numpy as np
import skimage.data
from smo.api import SMO

# Load image and mask saturated pixels
image = skimage.data.human_mitosis()
masked_image = np.ma.masked_greater_equal(image, 250)  # image is uint8 (0-255)

# Create an instance of the SMO operator
smo = SMO(sigma=1, size=7, shape=(1024, 1024))

# Calculate its background distribution
bg_rv = smo.bg_rv(masked_image)

# Calculate the median background
bg_rv.ppf(0.5)
```

## Development

Code style is enforced via pre-commit hooks. To set up a develoment environment, clone the repository, optionally create a virtual environment, install the [dev] extras and the pre-commit hooks:

```
git clone https://github.com/maurosilber/SMO
cd SMO
conda create -n smo python pip numpy scipy
pip install -e .[dev]
pre-commit install
```
