import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


INPUT_IMAGE_PATH = 'datasets/faces'
OUTPUT_CSV_FILE = 'datasets/faces.csv'
PROCESSED_IMAGE_PATH = 'datasets/processed'
PROCESSED_CSV_FILE = 'datasets/processed.csv'
DETECTED_FACE_PATH = 'datasets/cropped'
DETECTED_CSV_FILE = 'datasets/cropped.csv'
OUTPUT_MODEL_NAME = 'model.lib'
train_csv_file = PROCESSED_CSV_FILE

def train_model(train_csv, output_model_name):
    dataset = pd.read_csv(train_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]

    images = []
    print('Training recognition model ...')
    for item in names:
        image = cv2.imread(str(item), 0)
        resized = cv2.resize(image, (80,80), interpolation=cv2.INTER_LINEAR)
        images.append(np.ravel(resized))

    clf = SVC(kernel='poly', probability=True)
    clf.fit(images, labels)
    dump(clf, output_model_name)
    print('Model created in', output_model_name)
    input("Press [ENTER] key to continue...")

train_model(train_csv_file, OUTPUT_MODEL_NAME)