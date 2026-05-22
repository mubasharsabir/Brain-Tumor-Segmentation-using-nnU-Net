# 🧠 Brain Tumor Segmentation Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-00A6B4)

## 📖 Project Overview
This repository contains a full-stack, AI-powered web application designed to automate the volumetric segmentation of brain tumors from MRI scans. Built with a Flask backend and a deep learning engine powered by PyTorch and MONAI, it provides a seamless interface for medical professionals to upload patient scans and receive instant visual predictions.

The underlying model is a **3D U-Net** trained on the globally recognized **BraTS (Brain Tumor Segmentation)** dataset, enabling high-precision identification of complex tumor sub-regions.

## ✨ Key Features
* **Multi-Modal MRI Processing:** Simultaneously processes four distinct NIfTI modalities (`FLAIR`, `T1`, `T1CE`, `T2`) for accurate tissue differentiation.
* **Slice-by-Slice Visualization:** Allows users to input a specific Z-axis slice ID (0-154) to generate a 2D cross-sectional view with the segmentation mask overlaid.
* **Clinical-Grade AI Architecture:** Utilizes the MONAI framework's 3D U-Net optimized with Dice Loss for medical image segmentation.
* **Privacy-First Design:** Implements an ephemeral storage mechanism where uploaded patient files are processed in memory/temporary directories and strictly deleted post-inference.
* **Responsive Web Interface:** A clean, intuitive HTML/CSS frontend with dynamic loading states.

## 🏗️ System Architecture
1. **Frontend (Client):** HTML/CSS form accepts 4 `.nii.gz` files and a Slice ID.
2. **Backend (Server):** Flask (`app.py`) securely handles file uploads and triggers the AI pipeline.
3. **Data Pipeline:** `nibabel` reads the 3D NIfTI arrays, and `numpy` stacks them into a 4-channel tensor.
4. **Inference Engine:** PyTorch loads the `.pth` weights and passes the tensor through the 3D U-Net.
5. **Output Generation:** `matplotlib` renders the specific slice overlay, encodes it as a base64 string, and returns it to the frontend.

## 🛠️ Prerequisites
* **Operating System:** Windows, macOS, or Linux
* **Python:** Version 3.8 or higher
* **Hardware:** Minimum 8GB RAM. A CUDA-enabled NVIDIA GPU is highly recommended for faster inference, but CPU is fully supported.

## 🚀 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

### Step 2: Create a Virtual Environment (Recommended)
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install the required Python packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Add the Pre-Trained Model Weights
Ensure the trained weights file `brain_tumor_unet.pth` is placed in the root directory of the project (alongside `app.py`). Note: Due to file size limits, this file might not be included in the remote repository and needs to be downloaded separately.

## 💻 Usage Instructions

1. **Start the Web Server:** Execute the following command in your terminal:
   ```bash
   python app.py
   ```
2. **Access the Application:** Open your web browser and navigate to `http://127.0.0.1:5000/`.
3. **Run a Prediction:**
   * Upload the corresponding `FLAIR`, `T1`, `T1CE`, and `T2` `.nii.gz` files for a single patient.
   * Enter the desired **Z-Axis Slice ID** (e.g., `75`).
   * Click **Upload and Predict**. Wait a few moments for the model to process the 3D volume and render the result.

## 📂 Project Structure
```text
├── app.py                     # Main Flask application and backend routing
├── requirements.txt           # Python package dependencies
├── brain_tumor_unet.pth       # PyTorch model weights (required)
├── brain_tumor_unet.ipynb     # Jupyter notebook containing the model training code
├── templates/
│   └── index.html             # Frontend user interface
└── uploads/                   # Temporary directory for processing NIfTI files (auto-cleared)
```

## 🧠 Dataset Acknowledgment
This project was trained using the [BraTS (Brain Tumor Segmentation) Dataset](http://braintumorsegmentation.org/). We acknowledge the researchers and institutions that collected and annotated this critical medical data.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License, see the LICENSE file for details.
