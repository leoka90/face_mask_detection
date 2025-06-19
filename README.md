# Project Short Video Demo
![Short Demo (2)](https://github.com/user-attachments/assets/64749cc1-dd5e-4cdd-a641-dc84c32bc8cf)
![Short Demo (1)](https://github.com/user-attachments/assets/1c628069-cdfd-4699-b13c-44e352254d2e)


# Face Mask Detection with Live Alert System
A real-time deep learning-based face mask detection system that utilizes to detect the presence of a face mask on human faces on webcam video as well as on images and alert the authority to take action on those who are not wearing mask.

# Features
1.Real-time face detection using webcam
2.Binary classification: With Mask or No Mask
3.Alerts when a face without a mask is detected

# Tools and Libraries Used
1.Python 3.10 
2.TensorFlow/Keras — to build and train the AI model
3.OpenCV — for webcam video and face detection
4.NumPy — for handling image data
5.Haar Cascades — a simple face detection method
6.Playsound / OS module — for playing alert sounds

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



#Future Improvements
1.Add multi-face tracking with mask status per person
2.Support for video file input or IP camera
3.GUI dashboard for better monitoring
4.Integration with security systems for enterprise use





