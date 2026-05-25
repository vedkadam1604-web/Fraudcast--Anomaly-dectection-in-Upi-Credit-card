# Fraud & Anomaly Detection

A machine learning project to detect fraudulent credit card transactions and anomalous UPI payments using both supervised and unsupervised approaches.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced_Data-green)

## What This Does

Two fraud/anomaly detection systems in one app:

1. **Credit Card Fraud Detection** — flags suspicious card transactions based on amount, time, and behavioural patterns
2. **UPI Anomaly Detection** — detects unusual UPI payments based on velocity, location, beneficiary history, and timing

## Why This is Interesting

Financial fraud detection is one of the hardest ML problems because:
- Data is extremely imbalanced (~1-2% fraud)
- False positives annoy real customers, false negatives cost money
- Fraud patterns keep changing

To handle the imbalance I used **SMOTE** (Synthetic Minority Oversampling) and evaluated with Precision-Recall curves instead of just accuracy.

## Approach

### Credit Card
- Gradient Boosting classifier after SMOTE oversampling
- Isolation Forest for unsupervised anomaly scoring
- Features: transaction amount, time of day, 8 anonymized behavioural V-scores

### UPI Transactions
- Gradient Boosting classifier
- Features: amount, hour, day of week, merchant category, device type, location match, velocity, new beneficiary flag

## Project Structure

```
fraud-anomaly-detection/
├── data/
│   ├── creditcard.csv
│   └── upi_transactions.csv
├── notebooks/
│   └── fraud_analysis.py      # EDA + model training
├── models/
│   ├── cc_fraud_model.pkl
│   ├── cc_scaler.pkl
│   ├── upi_model.pkl
│   └── upi_scaler.pkl
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/vedkadam1604-web/Fraud-Anomaly-Detection.git
cd Fraud-Anomaly-Detection
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Live Demo

[Click here to try it](https://YOUR_APP_URL.streamlit.app)

## Disclaimer

Built for learning purposes using synthetic data modelled on real-world fraud patterns.
