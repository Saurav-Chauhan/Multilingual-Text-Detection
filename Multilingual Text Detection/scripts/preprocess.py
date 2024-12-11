import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Clean and preprocess data
    data.dropna(subset=['review'], inplace=True)  # Drop rows with missing reviews

    # Ensure all reviews and labels are valid
    data = data[data['review'].str.strip().astype(bool)]  # Remove empty reviews
    data = data[data['label'].isin(["positive", "neutral", "negative"])]

    # Map labels to integers
    data['label'] = data['label'].map({"positive": 2, "neutral": 1, "negative": 0})

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data