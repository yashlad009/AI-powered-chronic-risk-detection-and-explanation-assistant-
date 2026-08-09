"""
ANN Training and Evaluation Pipeline.
Reproducibly loads the dataset, cleans missing values, splits data with stratification,
normalizes features, trains the model with EarlyStopping/Dropout, saves artifacts,
and evaluates classification performance using multiple metrics.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 12. REPRODUCIBILITY: Set random seeds globally
random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)

# Add project root to sys.path to allow relative/absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from backend.app.ml.ann.model import ChronicRiskANN

def train_model():
    """Executes the complete and reproducible ANN training and evaluation pipeline."""
    
    # Define paths relative to the workspace root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    dataset_path = os.path.join(base_dir, "ml-training", "datasets", "diabetes.csv")
    saved_models_dir = os.path.join(base_dir, "ml-training", "saved_models")
    
    model_save_path = os.path.join(saved_models_dir, "ann_model.keras")
    scaler_save_path = os.path.join(saved_models_dir, "scaler.joblib")
    imputation_save_path = os.path.join(saved_models_dir, "imputation_values.joblib")
    plot_save_path = os.path.join(saved_models_dir, "learning_curves.png")

    os.makedirs(saved_models_dir, exist_ok=True)

    print("=========================================")
    print("STEP 1: DATA LOADING")
    print("=========================================")
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print("\nDataset Shape:")
    print(df.shape)
    
    print("\nColumns and Data Types:")
    print(df.dtypes)
    
    print("\nTarget Class ('Outcome') Distribution:")
    print(df["Outcome"].value_counts(normalize=True))
    print(df["Outcome"].value_counts())

    print("\n=========================================")
    print("STEP 2: INVALID ZERO HANDLING")
    print("=========================================")
    print("Replacing invalid 0s with NaN for: Glucose, BloodPressure, SkinThickness, Insulin, BMI...")
    cols_to_handle = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    # Make a copy to avoid warnings
    df_cleaned = df.copy()
    
    # Print zeros count before handling
    print("\nZero counts before replacement:")
    for col in cols_to_handle:
        print(f"  {col}: {(df_cleaned[col] == 0).sum()} zeros")

    df_cleaned[cols_to_handle] = df_cleaned[cols_to_handle].replace(0, np.nan)

    print("\nNaN counts after replacement:")
    print(df_cleaned.isnull().sum())

    print("\n=========================================")
    print("STEP 3: TRAIN/TEST SPLIT (STRATIFIED)")
    print("=========================================")
    print("Splitting the dataset into features (X) and target (y) with 80% train / 20% test...")
    X = df_cleaned.drop("Outcome", axis=1)
    y = df_cleaned["Outcome"]

    # Stratified split based on class labels to preserve target ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Train split shape: {X_train.shape}, Test split shape: {X_test.shape}")

    print("\n=========================================")
    print("STEP 4: IMPUTATION")
    print("=========================================")
    print("Computing mean/median from X_train only to prevent data leakage...")
    
    # Calculate values strictly from X_train
    imputation_values = {
        "Glucose": float(X_train["Glucose"].mean()),
        "BloodPressure": float(X_train["BloodPressure"].mean()),
        "BMI": float(X_train["BMI"].mean()),
        "SkinThickness": float(X_train["SkinThickness"].median()),
        "Insulin": float(X_train["Insulin"].median())
    }
    
    print(f"Computed imputation mapping: {imputation_values}")
    
    # Apply to X_train and X_test
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col, val in imputation_values.items():
        X_train[col] = X_train[col].fillna(val)
        X_test[col] = X_test[col].fillna(val)

    print("Imputation completed successfully (no NaNs remaining in train or test).")

    print("\n=========================================")
    print("STEP 5: FEATURE SCALING")
    print("=========================================")
    print("Scaling features using StandardScaler (fitted on X_train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n=========================================")
    print("STEP 6 & 7: ANN ARCHITECTURE & TRAINING")
    print("=========================================")
    # Initialize the architecture (Dense(16) -> Dropout(0.2) -> Dense(8) -> Dropout(0.2) -> Dense(1))
    model = ChronicRiskANN(input_dim=8, dropout_rate=0.2)

    # Set up EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    print("Training model with validation split=0.2...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\n=========================================")
    print("STEP 8: TRAINING VISUALIZATION")
    print("=========================================")
    print(f"Saving training history curves plot to: {plot_save_path}")
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss Curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history["loss"], label="Training Loss", color="#ff7f0e", linewidth=2)
    plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", color="#1f77b4", linewidth=2)
    plt.title("Model Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", color="#d62728", linewidth=2)
    plt.title("Model Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()

    print("\n=========================================")
    print("STEP 9 & 10: MODEL EVALUATION & PREDICTIONS")
    print("=========================================")
    # Evaluate model on untouched test set
    loss, accuracy = model.evaluate(X_test_scaled, y_test)
    
    # Generate probabilities and classes
    y_pred_probs = model.predict(X_test_scaled)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()

    # Calculate validation metrics
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)

    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nSample Predictions (First 10 Test Samples):")
    sample_df = pd.DataFrame({
        "Patient ID": range(1, 11),
        "Predicted Probability": y_pred_probs[:10].flatten(),
        "Predicted Class": y_pred[:10],
        "Actual Class": y_test[:10].values
    })
    print(sample_df.to_string(index=False))

    print("\n=========================================")
    print("STEP 11: MODEL SAVING")
    print("=========================================")
    print(f"Saving final trained model to: {model_save_path}")
    model.save(model_save_path)

    print(f"Saving standard scaler to: {scaler_save_path}")
    joblib.dump(scaler, scaler_save_path)

    print(f"Saving training-derived imputation values to: {imputation_save_path}")
    joblib.dump(imputation_values, imputation_save_path)

    print("\nAll pipeline tasks executed successfully.")
    
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "epochs": len(epochs_range),
        "train_accuracy": history.history["accuracy"][-1],
        "val_accuracy": history.history["val_accuracy"][-1],
        "model_path": model_save_path,
        "scaler_path": scaler_save_path,
        "imputation_path": imputation_save_path,
        "plot_path": plot_save_path
    }

if __name__ == "__main__":
    train_model()
