<h2>Repository structure</h2>
The project git repository consists of Python, XML, H5 files and Jupyter notebooks organised in Project and Experiment_results directories. The Project directory consists of files necessary for running the developed Facial Expression Recognition System. The main_window.py enables the camera capture, updates the GUI’s labels and is also able to recognise faces using the pre-trained haarcascade_frontalface_default.xml file with haar-like features and perform the emotion detection using the model.h5 file obtained from the neural network training. The ui_main_window.py file contains the GUI of the system. The recognition.py is an additional file capable of recognising multiple faces and showing the class probability of the fundamental facial expressions’ occurrences without the GUI wrapper. The Experiment_results directory consists of Jupyter Notebook files used for hyperparameter evaluation where the code used for training is saved along with the obtained results 

<h2>Running the system</h2>
To launch the system, go to the Project directory of the repository and run it using the following command: python3 main_window.py.

<h2>Instruction of usage</h2>
After launching the system, press the “Start” button to enable camera capture. Subsequently, the camera capture can be halted using the “Stop” button and resumed with the “Start” button.

<h2>Reused code</h2>
The reused code used for training the neural network comes from the following sources: <br>
YouTube: When Maths Meets Coding. Availabe at: https://www.youtube.com/watch?v=T3yR9DZT2mQ &#60;Accessed: 19/08/2021&#62; <br>
YouTube: Akshit Madan. Availabe at: https://www.youtube.com/watch?v=Bb4Wvl57LIk &#60;Accessed: 19/08/2021&#62; <br>
GitHub: Akmadan. Availabe at: https://github.com/akmadan/Emotion_Detection_CNN &#60;Accessed: 19/08/2021&#62; <br>
<br>
The reused code used for the GUI implementation comes from the following sources: <br>
YouTube: BERROUBA MOHAMED EL AMINE. Availabe at: https://www.youtube.com/watch?v=wSv4-Ne8e-Q &#60;Accessed: 19/08/2021&#62; <br>
GitHub: berrouba-med-amine. Availabe at: https://github.com/berrouba-med-amine/simple-python-camera-viewer-opencv3-PyQt5 &#60;Accessed: 19/08/2021&#62; 

<h2>Dataset</h2>
The model training and the conducted experiments have been performend using the Fer2013 and JAFFE datasets available here: <br>
Fer2013: https://www.kaggle.com/deadskull7/fer2013 &#60;Accessed: 19/08/2021 &#62; <br>
JAFFE: https://zenodo.org/record/3451524#.YTntptNKhQI &#60;Accessed: 19/08/2021 &#62; <br>