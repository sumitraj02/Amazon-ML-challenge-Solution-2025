# infer.py
import os
import joblib
import numpy as np
import pandas as pd
from features import extract_image_features

MODEL_DIR = "model"

def predict_price(catalog_content, image_name):
    """Predict price using saved multimodal model."""
    model_path = os.path.join(MODEL_DIR, "multimodal_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        raise FileNotFoundError("Model files not found. Please train first using train_multimodal.py")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Extract features
    X_text = vectorizer.transform([catalog_content]).toarray()
    image_path = os.path.join("images", image_name)
    X_img = extract_image_features([image_path])
    
    X_combined = np.hstack([X_text, X_img])
    price = model.predict(X_combined)[0]
    return float(round(price, 2))

if __name__ == "__main__":
    test = pd.read_csv("dataset/test.csv")
    test["price"] = test.apply(
        lambda r: predict_price(r["catalog_content"], r["image_name"]),
        axis=1
    )
    test[["sample_id", "price"]].to_csv("dataset/test_out.csv", index=False)
    print("Predictions saved to dataset/test_out.csv")
