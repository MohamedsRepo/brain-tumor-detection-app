Brain Tumor Detection App

Overview

The Brain Tumor Detection App is a Python-based tool that uses a convolutional neural network to identify and classify brain tumors from MRI images. It includes both training and inference components:

Training: Train or retrain the deep learning model on your dataset of labeled brain scans.

Inference: Launch a Streamlit web interface to upload MRI images and get real-time tumor detection results.

What’s Included

Data Folder (data/): Place your MRI image datasets here.

Models Folder (models/): Saving and loading trained model weights (e.g., .h5 files).

Training Script (trainer/traning_model.py): Script to train the CNN.

Web App (app/app.py): Streamlit interface for inference.

Configuration Files:

.gitignore for ignored files/folders

requirements.txt listing Python dependencies

Prerequisites

Python 3.8 or newer

Git to clone the repository

A virtual environment tool like venv or virtualenv

Installation

Clone the repository

git clone https://github.com/MohamedsRepo/brain-tumor-detection-app.git
cd brain-tumor-detection-app

Create and activate a virtual environment

python -m venv .venv
# PowerShell on Windows:
. .\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

Project Structure

brain-tumor-detection-app/
├── app/                         # Streamlit web interface
│   └── app.py                   # Launch point for inference
├── data/                        # MRI image datasets (ignored by Git)
├── models/                      # Trained model weights (ignored by Git)
├── trainer/                     # Training scripts
│   └── traning_model.py         # Train CNN on your data
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This documentation

Training the Model

Before running the web app, you must train the model and place the weights in models/:

python trainer/traning_model.py \
  --data data/your_dataset_folder \
  --output models/brain_tumor_model.h5

Running the App

Note: The app requires a trained model file (e.g., models/brain_tumor_model.h5).

streamlit run app/app.py

Visit http://localhost:8501 in your browser to upload MRI scans and view detection results.

Contributing

Feedback and pull requests are welcome! Please open an issue or submit a PR.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

Questions? Reach out at malshkeili01@gmail.com.

