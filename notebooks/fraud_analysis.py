import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature reference (human-readable)
MERCHANT_CATS = ['Food & Dining', 'Shopping', 'Electronics', 'Travel', 'Fuel', 'Healthcare']
TXN_TYPES     = ['Online', 'POS (In-Store)', 'ATM Withdrawal', 'Contactless']
CARD_TYPES    = ['Debit', 'Credit', 'Prepaid']

# ==============================================================
# PART 1 — CREDIT CARD FRAUD (Human-Readable Features)
# ==============================================================
print("=" * 60)
print("PART 1: Credit Card Fraud Detection")
print("=" * 60)

np.random.seed(42)
n_total = 10000
n_fraud = 150

# Normal transactions
normal = pd.DataFrame({
    'amount':          np.random.exponential(scale=80, size=n_total - n_fraud).clip(0.5, 5000),
    'hour':            np.random.choice(
                           np.concatenate([np.repeat(range(8, 21), 4), np.repeat(range(0, 8), 1), [21,22,23]]),
                           size=n_total - n_fraud),
    'merchant_cat':    np.random.choice(range(len(MERCHANT_CATS)), size=n_total - n_fraud,
                                        p=[0.30, 0.25, 0.15, 0.10, 0.12, 0.08]),
    'txn_type':        np.random.choice(range(len(TXN_TYPES)), size=n_total - n_fraud,
                                        p=[0.40, 0.35, 0.10, 0.15]),
    'card_type':       np.random.choice(range(len(CARD_TYPES)), size=n_total - n_fraud,
                                        p=[0.55, 0.40, 0.05]),
    'is_international':np.random.choice([0, 1], size=n_total - n_fraud, p=[0.92, 0.08]),
    'velocity':        np.random.poisson(lam=1.5, size=n_total - n_fraud).clip(0, 8),
    'is_new_merchant': np.random.choice([0, 1], size=n_total - n_fraud, p=[0.88, 0.12]),
    'mins_since_last': np.random.exponential(scale=180, size=n_total - n_fraud).clip(1, 1440),
    'class':           0
})

# Fraudulent transactions — different patterns
fraud = pd.DataFrame({
    'amount':          np.random.exponential(scale=400, size=n_fraud).clip(200, 8000),
    'hour':            np.random.choice([0, 1, 2, 3, 23, 22], size=n_fraud),
    'merchant_cat':    np.random.choice(range(len(MERCHANT_CATS)), size=n_fraud,
                                        p=[0.05, 0.30, 0.35, 0.20, 0.05, 0.05]),
    'txn_type':        np.random.choice(range(len(TXN_TYPES)), size=n_fraud,
                                        p=[0.60, 0.10, 0.20, 0.10]),
    'card_type':       np.random.choice(range(len(CARD_TYPES)), size=n_fraud,
                                        p=[0.20, 0.45, 0.35]),
    'is_international':np.random.choice([0, 1], size=n_fraud, p=[0.35, 0.65]),
    'velocity':        np.random.randint(5, 15, size=n_fraud),
    'is_new_merchant': np.random.choice([0, 1], size=n_fraud, p=[0.20, 0.80]),
    'mins_since_last': np.random.exponential(scale=10, size=n_fraud).clip(0.5, 60),
    'class':           1
})

cc_df = pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
cc_df['merchant_name'] = cc_df['merchant_cat'].map({i: c for i, c in enumerate(MERCHANT_CATS)})
cc_df['txn_type_name'] = cc_df['txn_type'].map({i: t for i, t in enumerate(TXN_TYPES)})
cc_df['card_name']     = cc_df['card_type'].map({i: c for i, c in enumerate(CARD_TYPES)})
cc_df.to_csv(os.path.join(DATA_DIR, 'creditcard.csv'), index=False)

print(f"Dataset: {cc_df.shape}, Fraud rate: {cc_df['class'].mean()*100:.2f}%")

# EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Credit Card Fraud - EDA', fontsize=14, fontweight='bold')

axes[0, 0].hist(cc_df[cc_df['class']==0]['amount'], bins=50, alpha=0.7, label='Normal', color='#2ecc71', density=True)
axes[0, 0].hist(cc_df[cc_df['class']==1]['amount'], bins=50, alpha=0.7, label='Fraud',  color='#e74c3c', density=True)
axes[0, 0].set_title('Amount Distribution')
axes[0, 0].set_xlabel('Amount (₹)')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 3000)

fraud_by_hour = cc_df[cc_df['class']==1].groupby('hour').size()
axes[0, 1].bar(fraud_by_hour.index, fraud_by_hour.values, color='#e74c3c', alpha=0.9)
axes[0, 1].set_title('Fraud Transactions by Hour')
axes[0, 1].set_xlabel('Hour of Day')
axes[0, 1].set_ylabel('Fraud Count')

intl_fraud = cc_df.groupby(['is_international', 'class']).size().unstack(fill_value=0)
intl_fraud.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'], edgecolor='white')
axes[1, 0].set_title('International vs Domestic')
axes[1, 0].set_xticklabels(['Domestic', 'International'], rotation=0)
axes[1, 0].legend(['Normal', 'Fraud'])

new_merch = cc_df.groupby(['is_new_merchant', 'class']).size().unstack(fill_value=0)
new_merch.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'], edgecolor='white')
axes[1, 1].set_title('New Merchant Flag')
axes[1, 1].set_xticklabels(['Known', 'New'], rotation=0)
axes[1, 1].legend(['Normal', 'Fraud'])

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'cc_eda.png'), dpi=150, bbox_inches='tight')
plt.show()
print("CC EDA saved.")

# Train
CC_FEATURES = ['amount', 'hour', 'merchant_cat', 'txn_type', 'card_type',
                'is_international', 'velocity', 'is_new_merchant', 'mins_since_last']

X = cc_df[CC_FEATURES]
y = cc_df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train_sc, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_bal, y_bal)

y_pred = rf_model.predict(X_test_sc)
y_prob = rf_model.predict_proba(X_test_sc)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

iso_forest = IsolationForest(contamination=0.015, random_state=42, n_estimators=100)
iso_forest.fit(X_train_sc)

joblib.dump(rf_model,     os.path.join(MODELS_DIR, 'cc_fraud_model.pkl'))
joblib.dump(iso_forest,   os.path.join(MODELS_DIR, 'iso_forest.pkl'))
joblib.dump(scaler,       os.path.join(MODELS_DIR, 'cc_scaler.pkl'))
joblib.dump(CC_FEATURES,  os.path.join(MODELS_DIR, 'cc_features.pkl'))
print("CC models saved.\n")

# ==============================================================
# PART 2 — UPI ANOMALY DETECTION (with location distance)
# ==============================================================
print("=" * 60)
print("PART 2: UPI Transaction Anomaly Detection")
print("=" * 60)

np.random.seed(99)
n_upi   = 8000
n_uanom = 200

categories = ['Food', 'Shopping', 'Recharge', 'Utilities', 'Transfer', 'Entertainment']
devices    = ['Android', 'iOS', 'Web']

upi_normal = pd.DataFrame({
    'amount':          np.random.exponential(scale=500, size=n_upi - n_uanom).clip(1, 20000),
    'hour':            np.random.choice(
                           np.concatenate([np.repeat(range(7, 22), 4), np.repeat(range(0, 7), 1)]),
                           size=n_upi - n_uanom),
    'day_of_week':     np.random.randint(0, 7, size=n_upi - n_uanom),
    'merchant_cat':    np.random.choice(range(len(categories)), size=n_upi - n_uanom),
    'device_type':     np.random.choice(range(len(devices)), size=n_upi - n_uanom),
    'location_match':  np.random.choice([0, 1], size=n_upi - n_uanom, p=[0.05, 0.95]),
    'txn_per_hour':    np.random.poisson(lam=1.5, size=n_upi - n_uanom).clip(0, 10),
    'new_beneficiary': np.random.choice([0, 1], size=n_upi - n_uanom, p=[0.85, 0.15]),
    'is_anomaly':      0
})

upi_anomaly = pd.DataFrame({
    'amount':          np.random.exponential(scale=8000, size=n_uanom).clip(5000, 100000),
    'hour':            np.random.choice([0, 1, 2, 3, 23], size=n_uanom),
    'day_of_week':     np.random.randint(0, 7, size=n_uanom),
    'merchant_cat':    np.random.choice(range(len(categories)), size=n_uanom),
    'device_type':     np.random.choice(range(len(devices)), size=n_uanom),
    'location_match':  np.random.choice([0, 1], size=n_uanom, p=[0.70, 0.30]),
    'txn_per_hour':    np.random.randint(8, 20, size=n_uanom),
    'new_beneficiary': np.random.choice([0, 1], size=n_uanom, p=[0.20, 0.80]),
    'is_anomaly':      1
})

upi_df = pd.concat([upi_normal, upi_anomaly], ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)
upi_df['merchant_name'] = upi_df['merchant_cat'].map({i: c for i, c in enumerate(categories)})
upi_df['device_name']   = upi_df['device_type'].map({i: d for i, d in enumerate(devices)})
upi_df.to_csv(os.path.join(DATA_DIR, 'upi_transactions.csv'), index=False)

print(f"UPI dataset: {upi_df.shape}, Anomaly rate: {upi_df['is_anomaly'].mean()*100:.2f}%")

UPI_FEATURES = ['amount', 'hour', 'day_of_week', 'merchant_cat', 'device_type',
                 'location_match', 'txn_per_hour', 'new_beneficiary']

X_upi = upi_df[UPI_FEATURES]
y_upi = upi_df['is_anomaly']

X_utr, X_ute, y_utr, y_ute = train_test_split(X_upi, y_upi, test_size=0.2, random_state=42, stratify=y_upi)
upi_scaler = StandardScaler()
X_utr_sc   = upi_scaler.fit_transform(X_utr)
X_ute_sc   = upi_scaler.transform(X_ute)

smote2 = SMOTE(random_state=42)
X_ubal, y_ubal = smote2.fit_resample(X_utr_sc, y_utr)

upi_model = RandomForestClassifier(n_estimators=100, random_state=42)
upi_model.fit(X_ubal, y_ubal)

y_upred = upi_model.predict(X_ute_sc)
print(f"Accuracy: {accuracy_score(y_ute, y_upred):.4f}, F1: {f1_score(y_ute, y_upred):.4f}")
print(classification_report(y_ute, y_upred, target_names=['Normal', 'Anomaly']))

joblib.dump(upi_model,    os.path.join(MODELS_DIR, 'upi_model.pkl'))
joblib.dump(upi_scaler,   os.path.join(MODELS_DIR, 'upi_scaler.pkl'))
joblib.dump(UPI_FEATURES, os.path.join(MODELS_DIR, 'upi_features.pkl'))

print("\nAll models saved! Run: cd app && streamlit run app.py")
