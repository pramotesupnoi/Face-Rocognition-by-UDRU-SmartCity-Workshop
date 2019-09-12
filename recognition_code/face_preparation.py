import cv2
import numpy as np
from pathlib import Path

OUTPUT_PATH = 'datasets/faces'
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,255,0)

label = input('input name: ')
output_path = Path(OUTPUT_PATH)
if not output_path.exists():
    output_path.mkdir()
output_face_path = Path(OUTPUT_PATH + '/' + label)
if not output_face_path.exists():
    output_face_path.mkdir()

capture = cv2.VideoCapture(0)
count = 0
n = 20
while count < n:
    ret, frame = capture.read()
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        ##face = image[y:y+h, x:x+w]

        cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)
        cv2.putText(image, 'count = ' + str(count) + ' of ' + str(n), (x,y-10), font, 0.6, color, thickness=2)
    cv2.imshow('face classifier', image)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        output_name = OUTPUT_PATH + '/' + label + '/img' + str(count) + '.jpg'
        cropped = frame[y:y+h, x:x+w]
        cv2.imwrite(output_name, cropped)
        count += 1
capture.release()
cv2.destroyAllWindows()
