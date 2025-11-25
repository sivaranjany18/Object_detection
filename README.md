 Project Overview

This project implements real-time object detection using YOLOv3, OpenCV, and your laptop camera.
The model detects objects from the COCO dataset (80 common objects such as person, car, bottle, etc.).

Features

Real-time webcam object detection

YOLOv3 deep learning model

Bounding boxes + confidence score

Auto-adjusted window size

Works on any system with webcam

 How It Works

Loads YOLOv3 network using OpenCV DNN module

Captures frames from your laptop camera

Preprocesses each frame into a YOLO-compatible blob

Performs forward pass through YOLO layers

Extracts bounding boxes, class names, and confidence

Displays the detection results in real time
