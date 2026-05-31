# Vietnamese Dishes Recognition 🍜

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.0-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![CNN](https://img.shields.io/badge/Deep_Learning-MobileNetV2-009688?style=for-the-badge)

</div>

## 📌 Introduction
Welcome to the **Vietnamese Dishes Recognition** project! This repository contains a deep learning computer vision model built to recognize and classify popular Vietnamese dishes from images. 

Using **Transfer Learning** on the highly efficient **MobileNetV2** architecture, this model is designed to be lightweight, fast, and highly accurate in identifying complex food imagery.

## 🚀 Features
- **Transfer Learning with MobileNetV2**: Leverages a pre-trained feature extractor from Google, fine-tuned specifically for Vietnamese cuisine.
- **Robust Image Preprocessing**: Integrated `tf.keras.layers.Rescaling` (normalizing pixels to `[-1, 1]`) and data caching/prefetching via `tf.data.AUTOTUNE` to achieve blazing-fast GPU training speeds.
- **Data Augmentation**: Enhances model generalization by applying random flips, rotations, and zooms during training to prevent overfitting.
- **Interactive UI (Colab)**: Features a built-in Jupyter widget interface allowing users to upload an image and get real-time dish predictions along with confidence scores.

## 📸 Dataset
The model was trained on a custom dataset of thousands of Vietnamese food images, covering classic dishes (e.g., Phở, Bánh mì, Cá kho, Thịt kho hột vịt, etc.). 
*Note: Due to size constraints, the raw image dataset is not included in this repository.*

## 🧠 Model Architecture
1. **Base Model**: MobileNetV2 (weights pre-trained on ImageNet).
2. **Global Average Pooling**: Reduces spatial dimensions to a single vector.
3. **Dropout Layer**: Rate = 0.2 to prevent overfitting.
4. **Classification Head**: Dense layer with Softmax activation corresponding to the number of dish classes.

## 💻 How to Use
1. Open the `Vietnamese Dishes Recognition.ipynb` notebook in Google Colab.
2. Ensure you have the dataset uploaded to your Google Drive and mount the drive.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Click **Run All** to train the model.
5. Scroll to the bottom to use the **Interactive Upload Widget** and test the AI with your own food photos!

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Built with ❤️ for Vietnamese Cuisine and AI.*
