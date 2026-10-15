"""
Builds the cleaned image set used in the notebook.

1. Download both zips from https://data.mendeley.com/datasets/j9sp322drp
   ("With Carbide.zip" and "Without Carbide.zip", about 8.7 GB together)
2. Put them in data/raw/ (zipped or unzipped, both work)
3. Run:  python prepare_data.py

data/source_files.csv lists the 1,362 original photos I kept (one per scene, no
augmented copies). This script copies only those files, resizes them to a quarter
of the original size (3024 x 4032 -> 756 x 1008) and saves them to data/processed/.
"""

import csv
import os
import sys
import zipfile

import cv2
import numpy as np

RAW_FOLDER = os.path.join("data", "raw")
OUT_FOLDER = os.path.join("data", "processed")
FILE_LIST = os.path.join("data", "source_files.csv")
SCALE = 4             # 3024 x 4032 becomes 756 x 1008
JPEG_QUALITY = 92


def open_source(folder_name):
    """Return a function that reads a file by name, from the zip or the unzipped folder."""
    zip_path = os.path.join(RAW_FOLDER, folder_name + ".zip")
    dir_path = os.path.join(RAW_FOLDER, folder_name)

    if os.path.isdir(dir_path):
        return lambda name: open(os.path.join(dir_path, name), "rb").read()
    if os.path.isfile(zip_path):
        z = zipfile.ZipFile(zip_path)
        return lambda name: z.read(folder_name + "/" + name)

    sys.exit(f"Could not find '{folder_name}.zip' or the '{folder_name}' folder in {RAW_FOLDER}/")


def main():
    with open(FILE_LIST, newline="") as f:
        rows = list(csv.DictReader(f))

    readers = {}
    for i, row in enumerate(rows, start=1):
        folder = row["source_folder"]
        if folder not in readers:
            readers[folder] = open_source(folder)

        out_path = os.path.join(OUT_FOLDER, row["file"])
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # decode, shrink with area interpolation (best for downsizing), save
        data = readers[folder](row["source"])
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if i % 100 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} images done")

    for label in ("carbide", "natural"):
        n = len(os.listdir(os.path.join(OUT_FOLDER, label)))
        print(f"{label}: {n} images")


if __name__ == "__main__":
    main()
