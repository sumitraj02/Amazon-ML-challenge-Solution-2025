# features.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch

def extract_text_features(texts, vectorizer=None, max_features=5000):
    """Convert text into TF-IDF vectors. Reuse existing vectorizer if provided."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        X_text = vectorizer.fit_transform(texts)
    else:
        X_text = vectorizer.transform(texts)
    return X_text, vectorizer

def extract_image_features(image_paths):
    """Extract image embeddings using pretrained ResNet."""
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = model(input_tensor).squeeze().numpy()
            features.append(feat)
        except:
            features.append(np.zeros(512))
    return np.array(features)
