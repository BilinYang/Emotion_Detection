import cv2 as cv 
import os

data_path = "adaboost/data/"

f_cascade = cv.CascadeClassifier("adaboost/haarcascade_frontalface_alt2.xml")

for img in os.listdir(data_path): 
    frame = cv.imread(data_path + img)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces: 
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv.imwrite(os.path.join('adaboost/result/', img), frame)