# 🧠 Brain Tumor Segmentation Web Application using nnU-Net

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-00A6B4)

**Repository:** [https://github.com/mubasharsabir/Brain-Tumor-Segmentation-using-nnU-Net](https://github.com/mubasharsabir/Brain-Tumor-Segmentation-using-nnU-Net)

## 📖 Project Overview
This repository contains a full-stack, AI-powered web application designed to automate the volumetric segmentation of brain tumors from MRI scans. Built with a Flask backend, the inference engine is powered by **nnU-Net**—a state-of-the-art, self-configuring deep learning framework for biomedical image segmentation. It provides a seamless interface for medical professionals to upload patient scans and receive instant visual predictions.

The underlying model is trained on the globally recognized **BraTS (Brain Tumor Segmentation)** dataset, enabling high-precision identification of complex tumor sub-regions.

## ✨ Key Features
* **Multi-Modal MRI Processing:** Simultaneously processes four distinct NIfTI modalities (`FLAIR`, `T1`, `T1CE`, `T2`) for accurate tissue differentiation.
* **Slice-by-Slice Visualization:** Allows users to input a specific Z-axis slice ID (0-154) to generate a 2D cross-sectional view with the segmentation mask overlaid.
* **State-of-the-Art Architecture:** Utilizes the robust `nnU-Net` framework, ensuring optimal preprocessing, network architecture, and post-processing for the BraTS dataset.
* **Privacy-First Design:** Implements an ephemeral storage mechanism where uploaded patient files and generated masks are processed in temporary directories and strictly deleted post-inference.
* **Responsive Web Interface:** A clean, intuitive HTML/CSS frontend with dynamic loading states.

## 🏗️ System Architecture
1. **Frontend (Client):** HTML/CSS form accepts 4 `.nii.gz` files and a Slice ID.
2. **Backend (Server):** Flask (`app.py`) securely handles file uploads and formats them for the `nnU-Net` pipeline.
3. **Data Pipeline:** Uploaded files are temporarily mapped to the required nnU-Net input format (`_0000.nii.gz`, `_0001.nii.gz`, etc.).
4. **Inference Engine:** The system triggers `nnUNetv2_predict` using the pre-trained model weights.
5. **Output Generation:** `matplotlib` renders the specific slice overlay from the resulting mask, encodes it as a base64 string, and returns it to the frontend.

## 🛠️ Prerequisites
* **Operating System:** Windows (via WSL), macOS, or Linux (Linux/WSL is highly recommended for nnU-Net)
* **Python:** Version 3.8 or higher
* **Hardware:** Minimum 8GB RAM. A CUDA-enabled NVIDIA GPU is **highly recommended** for nnU-Net inference.

## 🚀 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/mubasharsabir/Brain-Tumor-Segmentation-using-nnU-Net.git
cd Brain-Tumor-Segmentation-using-nnU-Net
```

### Step 2: Create a Virtual Environment
**On Linux / macOS / WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```
*(On Windows: `venv\Scripts\activate`)*

### Step 3: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```
*Note: Ensure `nnunetv2` is included in your environment.*

### Step 4: Configure nnU-Net Environment Variables
nnU-Net requires environment variables to locate the model weights. Set the `nnUNet_results` path in your terminal before running the app:
**On Linux/macOS:**
```bash
export nnUNet_results="/path/to/your/nnUNet_results"
```
*(Ensure you have downloaded and placed your trained BraTS model weights inside this directory).*

## 💻 Usage Instructions

1. **Start the Web Server:**
   Execute the following command in your terminal from the project root:
   ```bash
   python app.py
   ```
2. **Access the Application:**
   Open your web browser and navigate to `http://127.0.0.1:5000/`.
3. **Run a Prediction:**
   * Upload the corresponding `FLAIR`, `T1`, `T1CE`, and `T2` `.nii.gz` files for a single patient.
   * Enter the desired **Z-Axis Slice ID** (e.g., `75`).
   * Click **Upload and Predict**. The backend will process the files through `nnU-Net` and render the result.

## 📂 Project Structure
```text
├── .idea/                     # IDE configuration files
├── __pycache__/               # Compiled Python files
├── templates/
│   └── index.html             # Frontend user interface
├── uploads/                   # Temporary directory for incoming NIfTI files (auto-cleared)
├── predictions/               # Temporary directory for nnU-Net output masks (auto-cleared)
├── app.py                     # Main Flask application and backend routing
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```
*(Note: Ensure your `nnUNet_results` folder is securely stored either within the project or linked via environment variables).*

## 🧠 Dataset Acknowledgment
This project was trained using the [BraTS (Brain Tumor Segmentation) Dataset](http://braintumorsegmentation.org/). We acknowledge the researchers and institutions that collected and annotated this critical medical data.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the repository issues page if you want to contribute.

## 📄 License
This project is licensed under the MIT License, see the LICENSE file for details.
