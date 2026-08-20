# Waste Classification - Execution Guide

This guide explains how to set up and run the waste classification project.

## Project Structure

Ensure your project files are arranged as follows:

```text
waste_project/
│
├── dataset/             # Contains the image folders (cardboard, glass, etc.)
├── dataset.py
├── model.py
├── train.py
├── predict.py
└── requirements.txt
```

---

## Setup Environment

Open your terminal in the `waste_project` folder and run the following commands:

```bash
# 1. Create a virtual environment
python -m venv env

# 2. Activate the environment (Windows)
.\env\Scripts\activate

# 3. Activate the environment (Mac/Linux)
source env/bin/activate

# 4. Install required packages
pip install -r requirements.txt
```

---

## How to Run

### 1. Training the Model

To train the network and generate the saved model weights (`model.pth`), execute:

```bash
python train.py
```

### 2. Single Image Inference (Prediction)

To test the model on a specific image, pass the path of the image as an argument when running the script:

```bash
python predict.py path/to/your/image.jpg
```

**Example:**
If you have an image named `sample.png` in your root folder, run:

```bash
python predict.py sample.png
```

The script will output the predicted class directly in the terminal.
