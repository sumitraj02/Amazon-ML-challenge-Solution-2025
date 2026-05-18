import os
import pandas as pd
import numpy as np
import joblib
import torch
from features import extract_text_features, extract_image_features
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --- Custom Dataset for batching images ---
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img)
        except:
            # Return a zero image if loading fails
            return torch.zeros(3, 224, 224)

# --- Batch image feature extractor ---
def extract_image_features_batch(image_paths, batch_size=64, device='cuda'):
    dataset = ImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Pretrained ResNet18 without the final classification layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model(batch).squeeze(-1).squeeze(-1)  # shape: [batch, 512]
            all_features.append(feats.cpu().numpy())
    
    return np.vstack(all_features)

# --- Predictor function ---
def predictor(texts, image_paths, model, vectorizer, device='cuda'):
    # Text features
    X_text, _ = extract_text_features(pd.Series(texts), vectorizer=vectorizer)
    X_text = X_text.toarray()
    
    # Image features
    X_img = extract_image_features_batch(image_paths, device=device)
    
    # Combine features
    X_combined = np.hstack([X_text, X_img])
    
    # Predict
    prices = model.predict(X_combined)
    return prices

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Load test dataset
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test['image_path'] = test['image_name'].apply(lambda x: os.path.join('images', x))
    
    # Load trained model and vectorizer
    model = joblib.load('model/multimodal_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    
    # Make predictions in batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test['price'] = predictor(test['catalog_content'].tolist(),
                              test['image_path'].tolist(),
                              model, vectorizer, device=device)
    
    # Save predictions
    output_df = test[['sample_id', 'price']]
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"✅ Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
