# Guidelines for Creating Datasets

This document provides instructions for creating datasets for Tamil and Telugu using the `create_datasets.py` script.

## Prerequisites
- Ensure the English dataset is placed in the `data/` directory with the name `amazon_reviews_english.csv`.
- The dataset must contain the following columns:
  - `review`: The text of the review.
  - `label`: Sentiment label (`positive`, `neutral`, or `negative`).

## Steps
1. Run the `create_datasets.py` script to translate the English dataset into Tamil and Telugu.
   ```bash
   python scripts/create_datasets.py
   ```
2. The translated datasets will be saved as:
   - `amazon_reviews_tamil.csv`
   - `amazon_reviews_telugu.csv`

3. Verify the translated datasets for completeness and accuracy.