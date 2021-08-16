# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtCore import pyqtSignal, QThread

# import Opencv module
from keras.models import load_model
from time import sleep
import time
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

from ui_main_window import *

class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        #self.ui.retranslateUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)

        haarcascadePath = "haarcascade_frontalface_default.xml"
        modelPath = "model.h5"
        self.face_classifier = cv2.CascadeClassifier(haarcascadePath)
        self.classifier = load_model(modelPath)
        self.emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    # view camera
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray)

        if len(faces) > 0:
        #for (x,y,w,h) in faces:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (x, y, w, h) = faces
            cv2.rectangle(image2,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = self.classifier.predict(roi)[0]
                labelPredicted=self.emotion_labels[prediction.argmax()]

                cv2.putText(image2,labelPredicted,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

                angry = int(round(float(prediction[0])*100, 2))
                disgust = int(round(float(prediction[1])*100, 2))
                fear = int(round(float(prediction[2])*100, 2))
                happy = int(round(float(prediction[3])*100, 2))
                neutral = int(round(float(prediction[4])*100, 2))
                sad = int(round(float(prediction[5])*100, 2))
                surprise = int(round(float(prediction[6])*100, 2))

                self.ui.progressBarAngry.setProperty("value", angry)
                self.ui.progressBarDisgust.setProperty("value", disgust)
                self.ui.progressBarFear.setProperty("value", fear)
                self.ui.progressBarHappy.setProperty("value", happy)
                self.ui.progressBarNeutral.setProperty("value", neutral)
                self.ui.progressBarSad.setProperty("value", sad)
                self.ui.progressBarSurprise.setProperty("value", surprise)

                self.ui.labelAngry_2.setText(str(angry) + "%")
                self.ui.labelDisgust_2.setText(str(disgust) + "%")
                self.ui.labelFear_2.setText(str(fear) + "%")
                self.ui.labelHappy_2.setText(str(happy) + "%")
                self.ui.labelNeutral_2.setText(str(neutral) + "%")
                self.ui.labelSad_2.setText(str(sad) + "%")
                self.ui.labelSurprise_2.setText(str(surprise) + "%")

            else:
                cv2.putText(image2,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # get image infos
        height, width, channel = image2.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image2.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")

    def closeMainWindow(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())