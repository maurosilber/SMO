from pathlib import Path

import pandas as pd

from .methods import thresholding_methods

image_directory = Path("data/BBBC025/images")
summary_file = Path("data/BBBC025/BBBC025.csv")

# Plotting order
method_order = ["SMO", "cellpose", "stardist", *thresholding_methods]
channel_order = ["Hoechst", "PhGolgi", "Mito", "ERSytoBleed", "ERSyto"]


def file(num, channel, filename):
    """File location inside the repository."""
    relative_file = f"broad/hptmp/shsingh/image_dataset/{num}_{channel}/{filename}.tif"
    return image_directory / relative_file


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(summary_file)
    df["file"] = df.apply(lambda x: file(x.num, x.channel, x.filename), axis=1)
    return df
