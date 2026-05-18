# train_baseline.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from features import extract_text_features

def main():
    data = pd.read_csv("dataset/train.csv")
    
    X_text, vectorizer = extract_text_features(data["catalog_content"])
    y = data["price"]
    
    X_train, X_val, y_train, y_val = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print("Validation R^2:", model.score(X_val, y_val))
    
    joblib.dump(model, "text_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Baseline model saved!")

if __name__ == "__main__":
    main()
