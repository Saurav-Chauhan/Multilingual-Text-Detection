import pandas as pd
from googletrans import Translator
import os

translator = Translator()

def translate_reviews(input_file, output_file, target_language):
    data = pd.read_csv(input_file)
    if 'review' not in data.columns or 'label' not in data.columns:
        raise ValueError("Input file must contain 'review' and 'label' columns")

    translated_reviews = []
    for review in data['review']:
        try:
            translation = translator.translate(review, dest=target_language).text
            translated_reviews.append(translation)
        except Exception as e:
            print(f"Error translating review: {review}, Error: {e}")
            translated_reviews.append(None)

    data['translated_review'] = translated_reviews
    data = data.dropna(subset=['translated_review'])
    data.to_csv(output_file, index=False)
    print(f"Translated dataset saved to {output_file}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Translating reviews to Tamil...")
    translate_reviews(
        input_file="data/amazon_reviews_english.csv",
        output_file="data/amazon_reviews_tamil.csv",
        target_language="ta"
    )

    print("Translating reviews to Telugu...")
    translate_reviews(
        input_file="data/amazon_reviews_english.csv",
        output_file="data/amazon_reviews_telugu.csv",
        target_language="te"
    )