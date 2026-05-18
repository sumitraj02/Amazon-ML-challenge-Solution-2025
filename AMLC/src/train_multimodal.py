#tain_multimodal.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from features import extract_text_features, extract_image_features

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_on_batch(batch_file, model=None):
    print(f"\n🚀 Training on batch: {batch_file}")
    df = pd.read_csv(batch_file)

    # Ensure 'image_path' column exists
    df["image_path"] = df["image_name"].apply(lambda x: os.path.join("images", x))

    # --- TEXT FEATURES ---
    X_text, vectorizer = extract_text_features(df["catalog_content"])

    # --- IMAGE FEATURES ---
    X_img = extract_image_features(df["image_path"])

    # --- COMBINE TEXT + IMAGE FEATURES ---
    X_combined = np.hstack([X_text.toarray(), X_img])
    y = df["price"]

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    # If model not initialized, create new one
    if model is None:
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05)

    # Train model on current batch
    model.fit(X_train, y_train)
    print(f"✅ Batch Validation R²: {model.score(X_val, y_val):.4f}")

    return model, vectorizer


def main():
    model = None
    vectorizer = None

    # Train on each batch file sequentially
    for i in range(1, 4):
        batch_file = f"dataset/train_batch_{i}.csv"
        if os.path.exists(batch_file):
            model, vectorizer = train_on_batch(batch_file, model)
        else:
            print(f"⚠️ Skipping missing batch file: {batch_file}")

    # Save model and vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, "multimodal_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n🎯 Training complete! Model saved to `{MODEL_DIR}/`")

if __name__ == "__main__":
    main()
