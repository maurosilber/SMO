from ..BBBC025 import file

series = "37983"
channel = "Mito"
filenames = [
    "cellpaintingfollowup-reimage_f08_s8_w50c9e4dbf-b710-488a-94f8-fb769a829474",
    "cellpaintingfollowup-reimage_p12_s8_w5de88b323-cd60-41b6-bf43-c43c6eb87bf6",
    "cellpaintingfollowup-reimage_j02_s8_w5a7bc87d5-cf23-4619-8657-9a4f26456525",
]
files = [file(series, channel, filename) for filename in filenames]
