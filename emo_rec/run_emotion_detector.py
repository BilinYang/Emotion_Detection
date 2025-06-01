from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import argparse
import os
from keras.utils import disable_interactive_logging

disable_interactive_logging()

ap = argparse.ArgumentParser() 
ap.add_argument("-m", "--model", required = True,
                help = "name of emotion detector (needs to be already trained and built)")
ap.add_argument("-v", "--video", help = "path to the (optional) video file that we want to detect emotions from")
args = vars(ap.parse_args())

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

detector = cv2.CascadeClassifier("emo_rec/haarcascade_frontalface_default.xml")

model_name = args["model"].strip().lower().replace("_", "")
model_path = os.path.join("emo_rec/built_models", model_name + ".keras")
print("[INFO] loading model", model_name, "from the path" , model_path, "for emotion detection and recognition")
model = load_model(model_path)

if not args.get("video", False): 
    camera = cv2.VideoCapture(0)
    print("[INFO] trying to turn on your camera for emotion detection. Please navigate to your home page to see the camera and the detected emotion")
else: 
    camera = cv2.VideoCapture(args["video"])


fps = int(camera.get(cv2.CAP_PROP_FPS))
# output = cv2.VideoWriter('videos/output.avi')

success, frame = camera.read()
while success and (cv2.waitKey(1) & 0xFF) != ord('q'): 
    frame = imutils.resize(frame, width = 500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # initialize the canvas for visualization
    canvas = np.zeros((220, 320, 3), dtype = "uint8")
    # clone the frame so that we can draw on it 
    frame_clone = frame.copy()
    
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces)>0: 
        face = sorted(faces, key = lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse = True)[0]
        (x, y, x_w, y_h) = face
        roi = gray[y:y+y_h, x:x+x_w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype(float)/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)): 
            text = "{}: {:.2f}%".format(emotion, prob*100)

            # draw the label and probability bar on the canvas
            width = int(prob*200)
            cv2.rectangle(canvas, (5, 5+35*i), (width, 35*(i+1)-10), (0, 0, 255), -1) 
            cv2.putText(canvas, text, (width+5, 35*(i+1)-14), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.45, (255, 255, 255), 2)
        
        # draw the label (the predicted emotion) on the frame
        cv2.putText(frame_clone, label, (x + 5, y - 11), cv2.FONT_HERSHEY_COMPLEX, 
                    0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (x, y), (x+x_w, y+y_h), (0, 0, 255), 1)

        cv2.imshow("Face", frame_clone)
        cv2.imshow("Probabilities", canvas)
        

    success, frame = camera.read()



camera.release()
cv2.destroyAllWindows()










