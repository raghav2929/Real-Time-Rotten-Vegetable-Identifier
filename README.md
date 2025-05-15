# Real-Time Defective Vegetable Detection using Jetson Nano

A lightweight deep learning system for **real-time classification of fresh vs. stale vegetables** using MobileNetV2, deployed on NVIDIA Jetson Nano. This edge-AI solution enables offline, low-latency inference for smart agriculture and automated food quality control.

**Note:** This project is still in progress. Current inference speed and robustness are sufficient for demonstrations but **may not be production-ready**. Optimization steps like TensorRT conversion, pruning, quantization, and better dataset augmentation are recommended before deployment. Also note that if the model lags or causes fps drops, you can remove the finetuning and train it normally for a higher number of epochs



## Features

* MobileNetV2 pre-trained and fine-tuned on custom dataset
* Live classification via USB camera and OpenCV overlay
* Handles lighting variations, occlusions, and camera angle changes
* Real-time inference on Jetson Nano
* TensorFlow 2.5 and Python 3.6 optimized for edge deployment



## Demo

![demo\_screenshot](assets/result_fresh_apple.png)
![sample\_results](assets/result_rotten_banana.png)



## Installation

Installation instructions have been intentionally removed as this project is still under development. If you wish to run the code and experiment on your own, please review the source files and manually set up the environment accordingly. The setup instructions for the Jetson Nano are provided in the [Jetson Setup Guide](https://github.com/puravsood/vegetable-defect-detection/blob/main/jetson_setup.md).



## Usage

Usage instructions have been omitted since the system is not finalized. For experimental or educational use, please inspect the codebase in the `src/` directory.



## Project Structure

```
vegetable-defect-detection/
├── src/                  # Python source files
├── saved_models/         # Exported models(Basic and Fine tuned)
├── assets/               # Diagrams, videos, screenshots
├── jetson_setup.md       # Setup instructions for Jetson Nano
├── README.md
├── LICENSE
├── .gitignore
```



## Dataset

This project uses a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification). Special thanks to the dataset creator:

> **Dataset Name**: *Fresh and Stale Classification*
> **Creator**: *Swoyam*
> [View Dataset on Kaggle](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification)
> *(Used under the terms of the dataset’s license)*



## Results

* Accuracy: **>90%** on test set
* Inference time: \~1–2 seconds per frame
* Tested under variable lighting and occlusion conditions


## License

This project is licensed under the [MIT License](LICENSE).


## Tags

`#JetsonNano` `#EdgeAI` `#ComputerVision` `#TensorFlow` `#MobileNetV2` `#Python` `#AgriTech` `#OpenCV` `#TensorFLow 2.5` `#whl file` `#aarch64`
