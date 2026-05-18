# src/utils.py
import os
import requests
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse
import time

def download_images(csv_path, output_dir="images", update_csv=True, max_retries=3, delay=1.5):
    """
    Download product images from image_link column and save them locally.

    Parameters:
        csv_path (str): Path to the CSV file containing 'image_link' and 'sample_id'
        output_dir (str): Directory to save downloaded images
        update_csv (bool): If True, saves a new CSV with an added 'image_name' column
        max_retries (int): Maximum number of retries per image in case of failure
        delay (float): Delay (in seconds) between retries to avoid throttling

    Returns:
        DataFrame: Updated dataframe with 'image_name' column
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    image_names = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        image_url = row["image_link"]
        sample_id = row["sample_id"]

        # Generate filename
        image_name = os.path.basename(urlparse(image_url).path)
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_name = f"{sample_id}.jpg"
        image_path = os.path.join(output_dir, image_name)

        # Skip if already exists
        if os.path.exists(image_path):
            image_names.append(image_name)
            continue

        # Try to download
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    break
            except Exception as e:
                print(f"⚠ Retry {attempt+1}/{max_retries} failed for {image_url}: {e}")
                time.sleep(delay)
        else:
            print(f"❌ Failed to download after {max_retries} attempts: {image_url}")

        image_names.append(image_name)

    # Add new column
    df["image_name"] = image_names

    if update_csv:
        updated_path = csv_path.replace(".csv", "_with_images.csv")
        df.to_csv(updated_path, index=False)
        print(f"\n✅ Updated dataset saved to: {updated_path}")

    return df
