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


PROCESSED_IMAGE_PATH = 'datasets/processed'
PROCESSED_CSV_FILE = 'datasets/processed.csv'
OUTPUT_MODEL_NAME = 'model.lib'
train_csv_file = PROCESSED_CSV_FILE

def validate_model(validate_csv, model_name):
    dataset = pd.read_csv(validate_csv, sep=',')
    ids = dataset.values[:,0]
    names = dataset.values[:,1]
    labels = dataset.values[:,2]

    images = []
    print('Validating recognition model ...')
    for item in names:
        image = cv2.imread(str(item), 0)
        resized = cv2.resize(image, (80,80), interpolation=cv2.INTER_LINEAR)
        images.append(np.ravel(resized))

    clf = load(model_name)
    y_p = cross_val_predict(clf, images, labels, cv=3)
    print('Accuracy Score:', '{0:.4g}'.format(accuracy_score(labels,y_p) * 100), '%')
    print('Confusion Matrix:')
    print(confusion_matrix(labels,y_p))
    print('Classification Report:')
    print(classification_report(labels,y_p))

validate_model(train_csv_file, OUTPUT_MODEL_NAME)