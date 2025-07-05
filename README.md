# Drishti: Action Recognition AI using Custom Video Dataset

Drishti is a lightweight deep learning project for **human action recognition** in videos. The model is trained to recognize two distinct movements:
1. **Walking left/right**
2. **Going up/down stairs**

It uses a custom dataset, a Long-term Recurrent Convolutional Network (LRCN) architecture, and is designed for educational and experimental purposes in AI-based surveillance or monitoring systems.

---

## 🔗 Dataset

The dataset used for training the model is available [**here**](https://drive.google.com/drive/folders/1GdDySfSqiV0acUDMrKDo-UzIyws_-IqX).  
It contains categorized video clips for:
- **Walking**
- **Stairs (up/down)**

The videos are labeled and used for training an LRCN model to classify movement types.

---

## 🧠 Model Details

- **Architecture**: Long-term Recurrent Convolutional Network (LRCN)
  - Combines CNN for spatial feature extraction and LSTM for temporal understanding
- **Training File**: [`LRCN action recognition.ipynb`](./LRCN%20action%20recognition.ipynb)
- **Input Format**: Video frames extracted and processed from short clips
- **Output Classes**: `['walking', 'stairs up', 'stairs down']`

---

## 🛠 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/VaibhavPo/Drishti.git
cd Drishti
### 2. Install Requirements
Ensure Python 3.10.13 is used. Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
### 3. Train the Model
Open and run the Jupyter notebook:

bash
Copy
Edit
jupyter notebook "LRCN action recognition.ipynb"
This will generate the trained model file: download_model2.h5

### 4. Run Prediction
Use the main script to test predictions on new video inputs:

bash
Copy
Edit
python mainfile.py
Make sure download_model2.h5 is present in the same directory or provide the correct path in mainfile.py.


## Folder structure
Drishti/
├── dataset.txt                    # Link to dataset
├── LRCN action recognition.ipynb  # Training notebook
├── download_model2.h5             # Trained model (output of training)
├── mainfile.py                    # Main inference script
├── requirements.txt               # Dependencies list
└── README.md                      # Project description
