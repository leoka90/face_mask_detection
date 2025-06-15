# Face Mask Detection with Live Alert System
A real-time deep learning-based face mask detection system that utilizes to detect the presence of a face mask on human faces on live streaming video as well as on images and alert the authority to take action on those who'r not wearing mask.

# Features
Real-time face detection using live streaming video
Binary classification: With Mask or No Mask
Alerts when a face without a mask is detected

# Tools and Libraries Used
Python 3.10 
TensorFlow/Keras — to build and train the AI model
OpenCV — for webcam video and face detection
NumPy — for handling image data
Haar Cascades — a simple face detection method
Playsound / OS module — for playing alert sounds

# Dataset
link: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
We used a free dataset from Kaggle with two types of images:
with_mask — people wearing masks
without_mask — people not wearing masks

# How to Run
1. Prepare Dataset
   Download the dataset from Kaggle and place it inside a folder named dataset/
   split dataset into training and validation sets

2.Train the Model
  Run the training script to create mask_detector_model.h5

3.Start the detection script to activate the live streaming video and monitor mask status

# Alert System
 Plays an alert sound using playsound or os.system() when a "No Mask" face is detected





