from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import methods, process, segmentation, summary_file

tqdm.pandas()

all_methods = methods.get_all_methods()


def median_background(image, method: callable) -> int:
    mask = method(image)
    histogram = process.image_histogram(image[mask])
    median = process.median_from_histogram(histogram)
    return median


def process_file(file: Path) -> pd.Series:
    out = {}

    # File metadata
    num, channel = file.parent.stem.split("_")
    out["num"] = int(num)
    out["channel"] = channel
    out["filename"] = file.stem

    # Load image
    image = process.load_image(file)

    # Median background
    for name, method in all_methods.items():
        out[name] = median_background(image, method)

    # SMO mask size
    mask_smo = all_methods["SMO"](image)
    out["SMO_mask_size"] = mask_smo.sum() / mask_smo.size

    # Manual segmentation
    mask = segmentation.segment(image, *segmentation.manual_thresholds[channel])
    out["segmentation_size"] = (mask.size - mask.sum()) / mask.size

    return pd.Series(out)


files = Path("data/BBBC025/images").glob("**/37983_*/*.tif")
files = pd.Series(files)
medians = files.progress_apply(process_file)
medians = medians.set_index(["num", "channel", "filename"]).sort_index()
medians.to_csv(summary_file)
