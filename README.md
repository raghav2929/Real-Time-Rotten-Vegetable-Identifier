Real-Time Defective Vegetable Detection using Jetson Nano

A lightweight deep learning system for real-time classification of fresh vs. stale vegetables using MobileNetV2, deployed on NVIDIA Jetson Nano. This edge-AI solution enables offline, low-latency inference for smart agriculture and automated food quality control.

Note: This project is still in progress. Current inference speed and robustness are sufficient for demonstrations but may not be production-ready. Optimization steps like TensorRT conversion, pruning, quantization, and better dataset augmentation are recommended before deployment. Also note that if the model lags or causes fps drops, you can remove the finetuning and train it normally for a higher number of epochs

Features

MobileNetV2 pre-trained and fine-tuned on custom dataset
Live classification via USB camera and OpenCV overlay
Handles lighting variations, occlusions, and camera angle changes
Real-time inference on Jetson Nano
TensorFlow 2.5 and Python 3.6 optimized for edge deployment
