import os
import pandas as pd
import numpy as np
import joblib
from features import extract_text_features, extract_image_features

def predictor(sample_id, catalog_content, image_path, model, vectorizer):
    """
    Predict price using trained multimodal model.
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_path: Path to product image
    - model: Trained multimodal model
    - vectorizer: Fitted TF-IDF vectorizer for text features
    
    Returns:
    - price: Predicted price as a float
    """
    # Extract text features
    X_text, _ = extract_text_features(pd.Series([catalog_content]), vectorizer=vectorizer)
    
    # Extract image features
    X_img = extract_image_features([image_path])
    
    # Combine features
    X_combined = np.hstack([X_text.toarray(), X_img])
    
    # Predict price
    price = model.predict(X_combined)[0]
    
    return float(price)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Load test dataset
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Add image path column (assuming images are in 'images/' folder)
    test['image_path'] = test['image_name'].apply(lambda x: os.path.join('images', x))
    
    # Load trained model and vectorizer
    model = joblib.load('model/multimodal_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_path'], model, vectorizer),
        axis=1
    )
    
    # Select only required columns for submission
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"✅ Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
