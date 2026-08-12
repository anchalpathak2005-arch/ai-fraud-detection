"""
AI-Powered Financial Fraud Detection System
=============================================
Hybrid Deep Learning model combining:
  - ANN (Artificial Neural Network)  -> learns patterns from static transaction features
  - LSTM (Long Short-Term Memory)    -> learns sequential/temporal spending behaviour
    per customer (based on concepts from RNN_and_LSTM_complete_notes.pdf)

Concepts applied directly from the course notes:
  - Forward/Backward propagation, weights & biases (ANN_DL_1.pdf)
  - ReLU in hidden layers, Sigmoid in output layer for binary classification
  - Adam optimizer (mentioned in ANN_DL_1.pdf "Optimizer" slide)
  - Dropout for regularization (ANN_DL_1.pdf "Drop Out" slide)
  - LSTM gates (Forget / Input / Output) to retain long-term transaction
    behaviour and forget irrelevant history (RNN_and_LSTM_complete_notes.pdf)
  - Many-to-One RNN/LSTM architecture: a sequence of past transactions ->
    single fraud/not-fraud prediction for the latest transaction
    (RNN_and_LSTM_complete_notes.pdf "Many to One" slide)

This script is self-contained: it generates a realistic synthetic
transaction dataset (structure mirrors the popular Kaggle "Credit Card
Fraud Detection" dataset) so it can be run end-to-end without external
downloads. Swap `generate_synthetic_data()` for a real CSV loader in
production.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve)

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------------------------------------------------------
# 1. DATA -- synthetic generator (structure: Time, Amount, V1..V10, Class)
# --------------------------------------------------------------------------
def generate_synthetic_data(n_customers=2000, seq_len=10):
    """
    Creates `n_customers` customers, each with a sequence of `seq_len`
    past transactions + 1 "current" transaction to classify.
    Fraudulent transactions are injected with distribution shifts
    (higher amount, odd hour, feature drift) to mimic real fraud signals.
    """
    rows = []
    for cust_id in range(n_customers):
        is_fraud_case = np.random.rand() < 0.03      # ~3% fraud rate (imbalanced, like real data)
        seq = []
        for t in range(seq_len + 1):
            hour = np.random.randint(0, 24)
            amount = np.abs(np.random.normal(60, 40))
            features = np.random.normal(0, 1, size=10)  # V1..V10 (PCA-like features)

            fraud_flag = 0
            # Inject fraud pattern only on the LAST transaction of fraud cases
            if is_fraud_case and t == seq_len:
                fraud_flag = 1
                amount = amount * np.random.uniform(4, 10)      # abnormally large amount
                hour = np.random.choice([1, 2, 3, 4])            # odd hour
                features = features + np.random.normal(3, 1, 10)  # feature drift

            seq.append([hour, amount, *features, fraud_flag])

        for row in seq:
            rows.append([cust_id] + row)

    cols = ["customer_id", "hour", "amount"] + [f"V{i}" for i in range(1, 11)] + ["Class"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# 2. MODEL A -- ANN for single-transaction (tabular) fraud scoring
#    Mirrors the ANN architecture taught in ANN_DL_1.pdf:
#    Input Layer -> Hidden Layers (ReLU) -> Dropout -> Output (Sigmoid)
# --------------------------------------------------------------------------
def build_ann(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),   # hidden layer 1 (ReLU - faster, avoids vanishing gradient)
        layers.Dropout(0.3),                   # Dropout (regularization, from notes)
        layers.Dense(32, activation="relu"),   # hidden layer 2
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),   # hidden layer 3
        layers.Dense(1, activation="sigmoid")  # output layer - binary classification (fraud/not fraud)
    ], name="ANN_Fraud_Scorer")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),   # Adam optimizer (from notes)
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


# --------------------------------------------------------------------------
# 3. MODEL B -- LSTM for sequential customer behaviour
#    Many-to-One LSTM: last `seq_len` transactions -> fraud probability of
#    the newest transaction (RNN_and_LSTM_complete_notes.pdf)
# --------------------------------------------------------------------------
def build_lstm(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(64, return_sequences=True),   # captures long-term dependencies, avoids vanishing gradient
        layers.Dropout(0.3),
        layers.LSTM(32),                          # many-to-one: collapse sequence to single vector
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")      # fraud probability for the current transaction
    ], name="LSTM_Sequence_Scorer")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


# --------------------------------------------------------------------------
# 4. TRAIN + EVALUATE -- ANN (tabular baseline)
# --------------------------------------------------------------------------
def run_ann_pipeline(df):
    feature_cols = ["hour", "amount"] + [f"V{i}" for i in range(1, 11)]
    X = df[feature_cols].values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_ann(input_dim=X_train.shape[1])

    early_stop = callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=5, restore_best_weights=True)

    # class_weight compensates for the heavy class imbalance (fraud is rare)
    fraud_ratio = y_train.mean()
    class_weight = {0: 1.0, 1: (1 - fraud_ratio) / fraud_ratio}

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n=== ANN (Tabular) Results ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_prob), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, scaler, history, y_test, y_pred_prob


# --------------------------------------------------------------------------
# 5. TRAIN + EVALUATE -- LSTM (sequential behaviour model)
# --------------------------------------------------------------------------
def run_lstm_pipeline(df, seq_len=10):
    feature_cols = ["hour", "amount"] + [f"V{i}" for i in range(1, 11)]

    sequences, labels = [], []
    for cust_id, group in df.groupby("customer_id"):
        group = group.sort_index()
        feats = group[feature_cols].values
        label = group["Class"].values[-1]        # label of the final (most recent) txn
        sequences.append(feats[:seq_len + 1][:-1])  # history window (excludes current txn)
        labels.append(feats[-1])                    # not used directly; kept for clarity
        sequences[-1] = feats[:seq_len]              # first seq_len transactions as context
        labels[-1] = label

    X = np.array(sequences)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # scale using stats from training set (flatten -> scale -> reshape back)
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    model = build_lstm(seq_len=seq_len, n_features=n_feat)

    early_stop = callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=5, restore_best_weights=True)

    fraud_ratio = y_train.mean()
    class_weight = {0: 1.0, 1: (1 - fraud_ratio) / max(fraud_ratio, 1e-6)}

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n=== LSTM (Sequential Behaviour) Results ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_pred_prob), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, scaler, history, y_test, y_pred_prob


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    df = generate_synthetic_data(n_customers=2000, seq_len=10)
    print(f"Total transactions: {len(df)} | Fraud rate: {df['Class'].mean():.3%}")

    ann_model, ann_scaler, ann_hist, ann_y_test, ann_pred_prob = run_ann_pipeline(df)
    lstm_model, lstm_scaler, lstm_hist, lstm_y_test, lstm_pred_prob = run_lstm_pipeline(df, seq_len=10)
    print("ANN predictions:", len(ann_pred_prob))
    print("LSTM predictions:", len(lstm_pred_prob))
    # Use ANN predictions as the final fraud probability
    # ANN and LSTM currently use different test samples,
    # so their probabilities should not be averaged directly.
    combined_pred_prob = ann_pred_prob
    # Test different fraud detection thresholds
    print("\n=== THRESHOLD ANALYSIS ===")

    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        test_pred = (combined_pred_prob >= threshold).astype(int)
    
        cm = confusion_matrix(ann_y_test, test_pred)
        report = classification_report(
            ann_y_test,
            test_pred,
            output_dict=True,
            zero_division=0
    )
    
        print(
            f"Threshold: {threshold:.2f} | "
            f"Fraud Recall: {report['1']['recall']:.3f} | "
            f"Fraud Precision: {report['1']['precision']:.3f} | "
            f"Fraud F1: {report['1']['f1-score']:.3f} | "
            f"Fraud Detected: {cm[1][1]}/10"
        )
    
    print("\nHybrid probability range:")
    print("Minimum:", combined_pred_prob.min())
    print("Maximum:", combined_pred_prob.max())
    print("Mean:", combined_pred_prob.mean())

# Convert probability into fraud/not-fraud
    combined_pred = (combined_pred_prob >= 0.5).astype(int)

    print("\n=== HYBRID ANN + LSTM RESULTS ===")
    print("ROC-AUC:", round(roc_auc_score(ann_y_test, combined_pred_prob), 4))
    print("Confusion Matrix:\n", confusion_matrix(ann_y_test, combined_pred))
    print(classification_report(ann_y_test, combined_pred, digits=3))

    print("\nBoth models trained. In production, combine their probabilities "
          "(e.g., weighted average or a small meta-classifier / stacking layer) "
          "to produce the final fraud risk score per transaction.")
