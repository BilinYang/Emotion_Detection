import cv2 as cv 

# read image of a face
img = cv.imread('adaboost/face.jpg')
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)

# convert image from color to grayscale 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# load Haar Cascade Classifier 
f_cascade = cv.CascadeClassifier("adaboost/haarcascade_frontalface_alt2.xml")
faces = f_cascade.detectMultiScale(img)

for (x, y, w, h) in faces: 
    img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)