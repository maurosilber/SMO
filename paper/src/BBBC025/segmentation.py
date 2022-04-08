from skimage.morphology import binary_opening, disk

manual_thresholds = {
    "Mito": (1300, 1500),
    "ERSyto": (1000, 1250),
    "ERSytoBleed": (1400, 1600),
    "Hoechst": (520, 600),
    "PhGolgi": (4000, 5000),
}


def segment(image, high_threshold, low_threshold, disk_size=5):
    mask = binary_opening(image > low_threshold, disk(disk_size))
    mask |= image > high_threshold
    return mask
