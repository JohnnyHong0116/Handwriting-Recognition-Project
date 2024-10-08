# Handwriting Recognition Project

This project is designed to recognize and process handwritten input, including letters, digits, and mathematical symbols, using deep learning techniques. It integrates hand tracking and gesture control to allow users to write naturally on a virtual canvas, which is then processed in real-time by CNN models to recognize the content.

## Purpose

The main purpose of this project is to create an intuitive and efficient way to recognize and process handwritten content for educational, professional, and personal use. By leveraging real-time hand tracking and deep learning, the project aims to bridge the gap between traditional handwriting and digital text input, making it easier to digitize handwritten notes, solve mathematical expressions, and more.

## What the Project Does

- **Real-Time Handwriting Recognition**: The project captures handwritten input via a webcam and processes it using trained CNN models. The recognized text or symbols are displayed on a text screen in real-time.
- **Hand Tracking with Google MediaPipe**: The project uses Google MediaPipe's hand tracking technology to track the user's hand movements, allowing them to draw on a virtual canvas in the air. This enables natural and intuitive handwriting input without the need for a physical stylus or touch device.
- **Custom Dataset Collection**: Users can collect their own handwritten data using gesture controls, which can then be used to train or fine-tune the recognition models.
- **Support for Multiple Character Types**: The project supports recognition of English letters (A-Z), digits (0-9), and a wide range of mathematical symbols.

## Features

- **Custom Dataset Collection**: Capture and store custom handwritten datasets using gesture controls.
- **Letter Recognition**: Recognize handwritten letters (A-Z) using a CNN model.
- **Digit Recognition**: Recognize digits (0-9) using a pre-trained or custom-trained CNN model.
- **Math Symbol Recognition**: Recognize mathematical symbols using a custom-trained CNN model.
- **Real-Time Processing**: Perform real-time handwriting recognition using a camera feed with gesture control.
- **Text Output**: Display recognized characters and symbols in a text screen within the application.
- **Save to Word**: Automatically save the recognized text to a Word document before clearing the canvas.

## Datasets Used

- **Custom Letter Dataset**: A custom dataset created for recognizing English letters (A-Z). Each letter is stored in a separate folder within the dataset.
- **Custom Digit Dataset**: A custom dataset created for recognizing digits (0-9). Each letter is stored in a separate folder within the dataset.
- **CROHME Math Symbols Dataset**: A dataset used for training the model to recognize mathematical symbols. This dataset can be accessed from [CROHME(Kaggle source)](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols).
  
  **Note**: This project utilizes these datasets for educational and research purposes. The datasets are not included in this repository due to licensing restrictions. Please refer to the official sources to obtain them.

## Installation

### Prerequisites

To set up and run this project on your local machine, ensure you have the following prerequisites:

1. **Python 3.7 or later**
   - Install Python from the [official website](https://www.python.org/downloads/).

2. **Git**
   - Install Git for version control from the [official website](https://git-scm.com/downloads).

3. **pip (Python Package Installer)**
   - This is usually included with Python. You can check if you have it installed by running `pip --version` in your terminal.

4. **NVIDIA GPU with CUDA Support (Optional but recommended)**
   - If you have an NVIDIA GPU, you can leverage CUDA for faster processing.
   - Install the NVIDIA CUDA Toolkit from the [official website](https://developer.nvidia.com/cuda-downloads).
   - Install the appropriate version of cuDNN from the [NVIDIA Developer website](https://developer.nvidia.com/cudnn).
   - Verify your CUDA installation by running `nvcc --version` in your terminal.

5. **PyTorch with CUDA Support**
   - When installing [PyTorch](https://pytorch.org/), ensure you install the version that supports CUDA:
     ```bash
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
     ```
   - Replace `cu117` with the correct CUDA version you have installed.

6. **Virtual Environment (Optional but recommended)**
   - Create a virtual environment to manage your project dependencies separately:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows: `venv\Scripts\activate`
     - On macOS/Linux: `source venv/bin/activate`

7. **Required Python Packages**
   - Install all necessary Python packages using the `requirements.txt` file provided in the repository:
     ```bash
     pip install -r requirements.txt
     ```

8. **Webcam**
   - A functioning webcam is required for real-time hand tracking.

9. **Google MediaPipe**
   - Ensure MediaPipe is installed, which is used for hand tracking:
     ```bash
     pip install mediapipe
     ```

10. **Text Editor or IDE**
    - Use any text editor or IDE for Python development, such as VSCode, PyCharm, or Sublime Text.

11. **GitHub Account**
    - A GitHub account is necessary if you plan to contribute to the project or maintain your forked version.

### Additional Prerequisites

- **Microsoft Visual C++ Build Tools (Windows)**
  - If you encounter issues with building some Python packages on Windows, install the [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### Installation Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/JohnnyHong0116/Handwriting-Recognition-Project.git
    cd Handwriting-Recognition-Project
    ```

2. **Install Required Python Packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Datasets**:
    - Ensure that the required datasets (`custom_letter_dataset`, `Math Dataset`) are properly placed in the respective directories.
    - You can also train your models using the provided scripts (e.g., `letter_recognition.py`, `CROHME_Math_recognition.py`).

4. **Run the Application**:
    ```bash
    python main.py
    ```
    
## Usage

- **Custom Dataset Collection**: Use the custom dataset collection script to gather samples for training.
- **Model Training**: Train models using the provided scripts to recognize letters, digits, and mathematical symbols.
- **Real-Time Recognition**: Use the main application to recognize handwritten characters and symbols in real-time with gesture control.

## Project Structure

- **main.py**: Entry point for running the real-time handwriting recognition application.
- **letter_recognition.py**: Script for training a CNN model to recognize letters.
- **CROHME_Math_recognition.py**: Script for training a CNN model to recognize mathematical symbols.
- **custom_train_recognition.py**: Script for training a CNN model to recognize digits.
- **boundingbox_utils.py**: Utility functions for processing and saving bounding boxes during recognition.
- **Hand_Tracking.py**: Contains functions for gesture control and real-time recognition processing.
- **calibration.py**: Functions related to hand tracking calibration.
- **drawing_utils.py**: Utility functions for drawing bounding boxes and shapes.
- **geometry_utils.py**: Utility functions for calculating geometric properties.
- **gesture_detection.py**: Functions for detecting gestures.
- **smoothing.py**: Functions for smoothing gesture input.
- **hand_model.py**: Wrapper for the MediaPipe hand tracking model.
- **button_utils.py**: Utility functions for drawing and managing UI buttons.
- **utils.py**: General utility functions, including saving recognized text to Word documents.

## Models

- **Letter Recognition Model**: A CNN model trained on custom letter datasets.
- **Digit Recognition Model**: A CNN model trained on the MNIST dataset or a custom dataset.
- **Math Symbols Recognition Model**: A CNN model trained on the CROHME dataset for recognizing mathematical symbols.

## Sample Demo
![## Sample Demo](./Sample_Images/sample_img.jpg)

## Non-Commercial Use

This project is intended for educational and research purposes only. It is not intended for commercial use. Please ensure that you comply with the terms of use of the datasets and any third-party libraries used in this project.
