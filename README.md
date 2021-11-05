# SMO: robust and unbiased background distribution estimation by exploiting lack of local correlation

SMO is a Python package that implements the Silver Mountain Operator (SMO), which allows to recover an unbiased estimation of the background intensity distribution in a robust way.

## Installation

It can be installed with `pip` from PyPI:

```
pip install smo
```

## Plugins
### Napari

A [napari](https://napari.org) plugin is available.

To install:

- Option 1: in napari, go to `Plugins > Install/Uninstall Plugins...` in the top menu, search for `smo` and click on the install button.

- Option 2: just `pip` install this package in the napari environment.

It will appear in the `Plugins` menu.

### CellProfiler

A [CellProfiler](https://cellprofiler.org) plugin in available in the `smo/plugins/cellprofiler` folder.

![](images/CellProfiler_SMO.png)

To install, save [this file](https://raw.githubusercontent.com/maurosilber/SMO/main/smo/plugins/cellprofiler/smo.py) into your CellProfiler plugins folder. You can find (or change) the location of your plugins directory in `File > Preferences > CellProfiler plugins directory`.

### ImageJ / FIJI

An [ImageJ](https://imagej.net) plugin is available in the `smo/plugins/imagej` folder.

![](images/ImageJ_SMO.png)

To install, download [this file](https://raw.githubusercontent.com/maurosilber/SMO/main/smo/plugins/imagej/smo.py) and:

- Option 1: in the ImageJ main window, click on `Plugins > Install... (Ctrl+Shift+M)`, which opens a file chooser dialog. Browse and select the downloaded file. It will prompt to restart ImageJ for changes to take effect.

- Option 2: copy into your ImageJ plugins folder (`File > Show Folder > Plugins`).

To use the plugin, type `smo` on the bottom right search box:

![](images/ImageJ_MainWindow.png)

select `smo` in the `Quick Search` window and click on the `Run` button.

![](images/ImageJ_QuickSearch.png)

Note: the ImageJ plugin does not check that saturated pixels are properly excluded.

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

Code style is enforced via pre-commit hooks. To set up a development environment, clone the repository, optionally create a virtual environment, install the [dev] extras and the pre-commit hooks:

```
git clone https://github.com/maurosilber/SMO
cd SMO
conda create -n smo python pip numpy scipy
pip install -e .[dev]
pre-commit install
```
