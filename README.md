![PyPi](https://img.shields.io/pypi/pyversions/smo.svg)
[![License](https://img.shields.io/github/license/maurosilber/smo)](https://opensource.org/licenses/MIT)
[![PyPi](https://img.shields.io/pypi/v/smo.svg)](https://pypi.python.org/pypi/smo)
[![Conda](https://anaconda.org/conda-forge/smo/badges/version.svg)](https://anaconda.org/conda-forge/smo)
[![Paper](https://img.shields.io/badge/DOI-10.1364/JOSAA.477468-blue)](https://doi.org/10.1364/JOSAA.477468)

# SMO

SMO is a Python package that implements the Silver Mountain Operator (SMO),
which allows to recover an unbiased estimation of the background intensity distribution in a robust way.

## Citation

To learn more about the theory behind SMO,
you can read:

- the [peer-reviewed article](https://doi.org/10.1364/JOSAA.477468) in the Journal of the Optical Society of America,
- the [pre-print](https://doi.org/10.1101/2021.11.09.467975) in BioRxiv.

If you use this software,
please cite the peer-reviewed article.

## Usage

To obtain a background-corrected image, it is as straightforward as:

```python
import skimage.data
from smo import SMO

image = skimage.data.human_mitosis()
smo = SMO(sigma=0, size=7, shape=(1024, 1024))
background_corrected_image = smo.bg_corrected(image)
```

where we used a sample image from `scikit-image`.
By default,
the background correction subtracts the median value of the background distribution.
Note that the background regions will end up with negative values,
but with a median value of 0.

A notebook explaining in more detail the meaning of the parameters and other possible uses for SMO is available here: [examples/usage.ipynb](https://github.com/maurosilber/SMO/blob/main/examples/usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maurosilber/SMO/blob/main/examples/usage.ipynb).

## Installation

It can be installed with `pip` from PyPI:

```
pip install smo
```

or with `conda` from the conda-forge channel:

```
conda install -c conda-forge smo
```

## Development

To set up a development environment:

```
git clone https://github.com/maurosilber/SMO
cd SMO
pixi install
```

Code style is enforced via pre-commit hooks with `lefthook`.
