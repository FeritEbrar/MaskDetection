import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import time
import warnings
warnings.filterwarnings('ignore')

model = model_from_json(open("face_detection_feg", "r").read())
model.load_weights("weights_face_detection.h5")

face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

def preproccesing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    
    return img

def get_className(classNo):
    if classNo == 0:
        return "No Mask !"
    elif classNo == 1:
        return "Mask"

threshold = 0.6
initialTime = time.time()

while True:
    
    sucess, frame = cap.read()
    faces = face_detect.detectMultiScale(frame, 1.3, 5)
    
    for x,y,w,h in faces:
        
        roi = frame[y:y+h, x:x+h]
        img = cv2.resize(roi, (32,32))
        img = preproccesing(img)
        img = img.reshape(1,32,32,3)
        
        prediction = model.predict(img)
        classIndex = model.predict_classes(img)
        proba = np.amax(prediction)
        
        if proba > threshold:
            if classIndex==0:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(frame, (x,y-40),(x+w, y), (50,50,255),-2)
                cv2.putText(frame, "{} with % {:.2f}".format(str(get_className(classIndex)), proba),(x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255),1, cv2.LINE_AA)
    			
            elif classIndex==1:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(frame, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(frame, "{} with % {:.2f}".format(str(get_className(classIndex)), proba),(x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255),1, cv2.LINE_AA)
                    
        else:
            pass
        
        finishTime = time.time()
        fps = 1/(finishTime-initialTime)*100
        cv2.putText(frame, str(fps), (500,40), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),1, cv2.LINE_AA)
        
    cv2.imshow("Mask Detection", frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()    








