from pathlib import Path

import pandas as pd

from .methods import thresholding_methods

image_directory = Path("data/BBBC025/images")
summary_file = "data/BBBC025/BBBC025.csv"

# Plotting order
method_order = ["SMO", "cellpose", "stardist", *thresholding_methods]
channel_order = ["Hoechst", "PhGolgi", "Mito", "ERSytoBleed", "ERSyto"]


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(summary_file)
    df.file = df.file.apply(lambda file: image_directory / file)
    return df
