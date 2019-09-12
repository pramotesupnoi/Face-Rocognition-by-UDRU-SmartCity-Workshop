import cv2
import numpy as np
import pandas as pd
from pathlib import Path


HAAR_MODEL = 'haarcascade_frontalface_default.xml'

INPUT_IMAGE_PATH = 'datasets/faces'
OUTPUT_CSV_FILE = 'datasets/faces.csv'

def create_csv(dataset_path, output_csv):
    root_dir = Path(dataset_path)
    items = root_dir.iterdir()

    filenames = []
    labels = []
    print('reading image files ... ')
    for item in items:
        if item.is_dir():
            for file in item.iterdir():
                if file.is_file():
                    print(str(file))
                    filenames.append(file)
                    labels.append(item.name)
    raw_data = {'filename': filenames, 'label': labels}
    df = pd.DataFrame(raw_data, columns=['filename','label'])
    df.to_csv(output_csv)
    print(len(filenames), 'image file(s) read')
    input("Press [ENTER] key to continue...")

create_csv(INPUT_IMAGE_PATH, OUTPUT_CSV_FILE)