---
title: AI Voice Support Console
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# Voice-Enabled Customer Support Ticket Classification System

An end-to-end AI project that allows customer support inputs via speech or text, transcribes them using **Whisper**, automatically extracts sentiment and issue priority, and classifies the tickets using a Machine Learning pipeline (TF-IDF + Logistic Regression). 

The model is trained dynamically using the Kaggle dataset:
`mirzayasirabdullah07/customer-support-tickets-dataset-200k-records`

## Setup Instructions

1. **Navigate to the Project Directory**
Ensure you are in the directory where the project is saved:
```bash
cd /Users/yashna/NLPLABEL/voice_ticket_system
```

2. **Install the Required Python Packages**
It's highly recommended to use a virtual environment, but either way, install the necessary dependencies using `pip`:
```bash
pip install -r requirements.txt
```

*(Note: Depending on your system and Python version, you might need to install `ffmpeg` or `rust` if Whisper configuration fails out-of-the-box. On macOS run `brew install ffmpeg`)*

3. **Running the Pipeline End-to-End**

**Option 1: Web Interface Dashboard (Streamlit)**
You can directly run the Streamlit UI. If the model is not trained yet, it will automatically download the kaggle dataset (a sample for performance) and train the classifier before giving you access to the dashboard.
```bash
streamlit run app.py
```

**Option 2: Offline Explicit Training**
If you want to train the model offline on a larger chunk of the dataset (without the Streamlit UI overhead), run the training pipeline first:
```bash
python train_pipeline.py
```
This script saves `model.pkl` and `vectorizer.pkl`. Once training finishes successfully, run `streamlit run app.py`.

## Built-In Features
- **Auto-Kaggle Sync:** Automatically downloads the dataset natively and detects dynamic column mappings.
- **Natural Language Toolkit Operations:** Handles Regex cleaning, SW removal, lemmatization and auto tokenization.
- **Support Intent ML Classifier:** Baseline Logistic Regression wrapped in an optimized 5-category layout (Billing/Tech/Refund/Account/General)
- **Sentiment Engine:** Predicts Positivity & Priority based on conversational tone.
- **Whisper Connect:** Out of the box conversion from voice queries to intent via local ML layers.

Enjoy your Voice AI application!
