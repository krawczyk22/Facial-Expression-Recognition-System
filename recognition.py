from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

haarcascadePath = "haarcascade_frontalface_default.xml"
modelPath = "model.h5"
face_classifier = cv2.CascadeClassifier(haarcascadePath)
classifier =load_model(modelPath)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            labelPredicted=emotion_labels[prediction.argmax()]

            cv2.putText(frame,labelPredicted,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv2.putText(frame,"Surprised " + str(round(prediction[6]*100, 2)),(x,y-35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Sad " + str(round(prediction[5]*100, 2)),(x,y-65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Neutral " + str(round(prediction[4]*100, 2)),(x,y-95),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Happy " + str(round(prediction[3]*100, 2)),(x,y-125),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Fear " + str(round(prediction[2]*100, 2)),(x,y-155),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Disgust " + str(round(prediction[1]*100, 2)),(x,y-185),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,"Angry " + str(round(prediction[0]*100, 2)),(x,y-215),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()