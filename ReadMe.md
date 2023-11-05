# ML Assignment - FaceFirst
This repository contains implementation for the problem statement stated below. This work is been done as past of practical assignment for the role of Machine Learning Engineer.

## Problem Statement
**Detection**:

* Detect Faces, masked faces, faces with sunglasses and faces with both mask and sunglasses in Images and Videos.
* Crop them and segregate the images into respective folders (manual process...using python code to do this)
Note: If you can do it in video streams that would be great.
 Classifier:

**Requirement**: Model size should be 15 MB; Accuracy should be above 90% and inference time should be less than 6 milli seconds.      

Design a four-class classifier (SVM, Resenet or any) using deep learning models, which classifies the face images as
* Faces
* Masked faces
* Faces with sunglasses
* Masked faces with sunglasses

Use any dataset which is publicly available for training, validation and testing the four-class classifier network given in problem 2.    

**Testing**:
Use completely different set of images/video stream which are not used in the Classifier problem.

Test your model and see what kind of accuracy you are getting?

**Quantization**:
Reducing the final model size (e.g., 32 bit to 16 bit or 16 bit to 8 bit)
* What is the impact on accuracy?
* Impact on Inference?

**ONNX Conversion**:
Convert the final model to ONNX          

## Implemented models
**Classification**
* ResNet18
* EfficientNet_b0
* MobileNet_v2_050

Before passing data to the classifier, images are passed through face detection method to capture only faces frame from the images.

In this implementation I have used MTCNN as face detector.

**Object Detection**
* Faster-RCNN
* YOLOv5

## Framework
All models are written in pytorch first and later converted to ONNX format.

Key results and observation along with challanges faced while completing the assignment is mentioned in the attach PPT `Facefirst.pdf`.

The required files such as saved model and train data necessary for the assignments are available [here](https://drive.google.com/drive/folders/1tDve6dpwejNpakrFJv5Y8yg7a3_HKSw4?usp=drive_link).