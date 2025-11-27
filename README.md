# 🎤 Speech Emotion Recognition Using Spectral Features and Deep Learning
SIT789 – Robotics, Computer Vision and Speech Processing  
**Distinction Task 8.2**

This project implements a complete **Speech Emotion Recognition (SER)** pipeline using:
- **Handcrafted spectral features** (SC, SBW, SBE)  
- **Classical machine learning (SVM)**  
- **Deep learning approaches** (1D-CNN & 2D-CNN)

It follows the requirements of Deakin University's SIT789 Task 8.2.

---

## 📌 Project Overview

This project investigates how different audio representations and modelling approaches affect emotion classification accuracy.  
Given speech clips from four emotion categories:

- **Angry**
- **Calm**
- **Happy**
- **Sad**

the following methods are implemented:

### ✅ 1. Handcrafted Spectral Features  
- **Spectral Centroid (SC)**  
- **Spectral Bandwidth (SBW)**  
- **Spectral Band Energy (SBE)**  

These features are computed **manually (no librosa.feature SC/SBW/SBE allowed)** using mel-scale spectrograms with:
- `n_mels = 3401`
- `n_fft = 16384`
- `hop_length = 15 ms`

### ✅ 2. Speech Emotion Recognition using SVM  
SVM classifier (RBF kernel, C=10) is trained on SC, SBW, and SBE feature vectors.  
For each feature type:
- Overall accuracy  
- Confusion matrix  
are reported.

### ✅ 3. Deep Learning Approaches  
Two neural architectures are developed:

#### 🔹 3.1 1D-CNN on Raw Audio  
- Custom PyTorch Dataset (`EmotionSpeechDataset`)  
- 1D convolutional, pooling, dropout, and FC layers  
- Input = 1D waveform (normalized + padded/truncated to fixed length)  
- Learning curve and confusion matrix included  

#### 🔹 3.2 2D-CNN on Mel-Spectrogram Images  
- Mel-scale spectrograms saved as grayscale images  
- Images resized (e.g., 150×50)  
- 2D CNN with convolutional, pooling, dropout, and FC layers  
- Results compared with SVM and 1D-CNN  

---

## 📂 Repository Structure

