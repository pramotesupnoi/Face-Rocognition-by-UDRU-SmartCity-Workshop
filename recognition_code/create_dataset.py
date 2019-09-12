import cv2
import numpy as np
import pandas as pd
from pathlib import Path

HAAR_MODEL = 'haarcascade_frontalface_default.xml'

INPUT_IMAGE_PATH = 'datasets/faces'
OUTPUT_CSV_FILE = 'datasets/faces.csv'
PROCESSED_IMAGE_PATH = 'datasets/processed'
PROCESSED_CSV_FILE = 'datasets/processed.csv'


DETECT_FACE = False
train_csv_file = PROCESSED_CSV_FILE
def resize(image, width=None, height=None):
    (h, w) = image.shape[:2]
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    return cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

def process_image(input_csv, output_csv, output_path_name):
    dataset = pd.read_csv(input_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]

    output_path = Path(output_path_name)
    if not output_path.exists():
        output_path.mkdir()

    filenames = []
    print('preprocessing images ... ')
    for item in names:
        input_path = Path(item)
        if input_path.is_file():
            output_name = output_path_name + '/image' + str(ids[len(filenames)]) + input_path.suffix
            print(input_path.suffix)
            print(input_path, '->', output_name)
            image = cv2.imread(str(input_path))
            image = resize(image, width=256)
            cv2.imwrite(output_name, image)
            filenames.append(output_name)
    prc_data = {'filename': filenames, 'label': labels}
    df = pd.DataFrame(prc_data, columns=['filename', 'label'])
    df.to_csv(output_csv)
    print(len(filenames), 'image file(s) processed')
    input("Press [ENTER] key to continue...")

process_image(OUTPUT_CSV_FILE, PROCESSED_CSV_FILE, PROCESSED_IMAGE_PATH)