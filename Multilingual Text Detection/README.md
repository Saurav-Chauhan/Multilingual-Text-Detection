# Multilingual Text Sentiment Detection

This project focuses on leveraging multilingual pre-trained language models (MPLMs) and prompt engineering to improve sentiment detection for low-resource languages like Tamil and Telugu using the Amazon Review dataset.

## Features
- Supports Tamil, Telugu, and English.
- Uses XLM-RoBERTa for multilingual sentiment analysis.
- Preprocessing, training, and inference scripts included.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place datasets in the `data/` directory.

## Usage
- Preprocess data: `python scripts/preprocess.py`
- Train model: `python scripts/train.py`
- Infer sentiment: `python scripts/infer.py`
- Create datasets: `python scripts/create_datasets.py`