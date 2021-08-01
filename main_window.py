# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, QObject
from PyQt5.QtCore import pyqtSignal

# import Opencv module
from keras.models import load_model
from time import sleep
import time
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

from ui_main_window import *

class DummyThread(QObject):
    finished = pyqtSignal()
    def run(self):
        time.sleep(0)
        self.finished.emit()

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

        for (x,y,w,h) in faces:
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

                thread = DummyThread(self)
                thread.start()
                thread.finished.connect(lambda : self.changeLabels(prediction))

            else:
                cv2.putText(image2,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                self.ui.progressBarAngry.setProperty("value", 0)
                self.ui.progressBarDisgust.setProperty("value", 0)
                self.ui.progressBarFear.setProperty("value", 0)
                self.ui.progressBarHappy.setProperty("value", 0)
                self.ui.progressBarNeutral.setProperty("value", 0)
                self.ui.progressBarSad.setProperty("value", 0)
                self.ui.progressBarSurprise.setProperty("value", 0)

                self.ui.labelAngry_2.setText("Form", "0", 2)
                self.ui.labelDisgust_2.setText("Form", "0", 2)
                self.ui.labelFear_2.setText("Form", "0", 2)
                self.ui.labelHappy_2.setText("Form", "0")
                self.ui.labelNeutral_2.setText("Form", "0")
                self.ui.labelSad_2.setText("Form", "0")
                self.ui.labelSurprise_2.setText("Form", "0")

        # get image infos
        height, width, channel = image2.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image2.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

    def changeLabels(self, labels):
        self.ui.progressBarAngry.setProperty("value", round(labels[0]*100, 2))
        self.ui.progressBarDisgust.setProperty("value", round(labels[1]*100, 2))
        self.ui.progressBarFear.setProperty("value", round(labels[2]*100, 2))
        self.ui.progressBarHappy.setProperty("value", round(labels[3]*100, 2))
        self.ui.progressBarNeutral.setProperty("value", round(labels[4]*100, 2))
        self.ui.progressBarSad.setProperty("value", round(labels[5]*100, 2))
        self.ui.progressBarSurprise.setProperty("value", round(labels[6]*100, 2))

        #self.ui.labelAngry_2.setText("Form", str(round(labels[0]*100, 2)))
        #self.ui.labelDisgust_2.setText("Form", str(round(labels[1]*100, 2)))
        #self.ui.labelFear_2.setText("Form", str(round(labels[2]*100, 2)))
        #self.ui.labelHappy_2.setText("Form", str(round(labels[3]*100, 2)))
        #self.ui.labelNeutral_2.setText("Form", str(round(labels[4]*100, 2)))
        #self.ui.labelSad_2.setText("Form", str(round(labels[5]*100, 2)))
        #self.ui.labelSurprise_2.setText("Form", str(round(labels[6]*100, 2)))

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


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())