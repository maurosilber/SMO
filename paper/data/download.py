"""Downloads images from the Broad BioBenchmark Collection.

It downloads only the series 37983 of the BBBC025 dataset,
which consists of 5 tar.gz files, one for each channel,
32 GB in total.

They are extracted to data/images/BBBC025.
"""
import argparse
import tarfile
from urllib.request import urlopen

from tqdm import tqdm

base_url = "https://data.broadinstitute.org/bbbc/BBBC025/DatasetS8/"
files = [
    "37983_ERSyto.tar.gz",
    "37983_ERSytoBleed.tar.gz",
    "37983_Hoechst.tar.gz",
    "37983_Mito.tar.gz",
    "37983_PhGolgi.tar.gz",
]


def download_and_extract(url, *, N_images: int = None):
    TOTAL_IMAGES = 3456

    with urlopen(url) as file:

        if N_images is None:
            total = file.length
        else:
            total = file.length * N_images / TOTAL_IMAGES

        with tqdm.wrapattr(file, "read", total=total, miniters=1, leave=False) as file:
            with tarfile.open(fileobj=file, mode="r:gz") as tar:
                if N_images is None:
                    tar.extractall("data/images/BBBC025")
                    return

                for _ in tqdm(range(N_images), leave=False):
                    tar.extract(tar.next(), path="data/BBBC025/images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_images", type=int, default=None, action="store")
    args = parser.parse_args()

    for filename in tqdm(files):
        download_and_extract(base_url + filename, N_images=args.N_images)
