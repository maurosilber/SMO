# Paper

## Code to reproduce article

Steps to reproduce assume you have `git` and the `conda` package manager installed.

### Set up the environment

1. Clone this branch with:

```
git clone -b paper --depth 1 https://github.com/maurosilber/SMO.git
```

2. Move into the paper directory:

```
cd SMO/paper
```

3. Create a conda envionment and install all required Python packages with:

```
conda env create --file environment.lock.yaml
```

(or `conda env create --file environment.yaml`)

4. Activate the conda environment:

```
conda activate smo-paper
```

5. Install the SMO package:

```
pip install ..
```

*Note: it is 2 dots, as it is in the parent directory.*

### Download and run analysis

*Note: this can be skipped to reproduce the figures.
Go to "Generate figures" below.*

6. Download images (32 GB):

```
python data/download.py
```

7. Run analysis (72 hs on a 16-core CPU):

```
python -m src.BBBC025
```

*Note: most of the time is due to stardist and cellpose,
which can be sped up with a GPU.*

### Generate figures

If the previous step was skipped,
you need to download a subset to reproduce some figures:

To reproduce figures 5 and 6,
download the first 50 images (600 MB) with:

```
python data/download --N_images 50
```

 To reproduce supplementary figure 8,
 you might need to download the full dataset.

8. Generate all figures with:

```
python -m src
```

Add `--save` to export the figures to the `figure/` directory:

```
python -m src --save
```

To generate only a particular figure,
run its submodule.

For figures 1 to 6, run:
`python -m src.figure{i}`,
replacing `{i}` with the figure number.

For instance, for figure 1:

```
python -m src.figure1
```

For supplementary figures 1 to 8,
run: `python -m src.suppfig{i}`
