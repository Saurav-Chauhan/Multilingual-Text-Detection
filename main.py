import argparse
from scripts.train import train_model
from scripts.infer import SentimentInference
from scripts.create_datasets import translate_reviews
from utils.helpers import ensure_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Sentiment Analysis")
    parser.add_argument("mode", choices=["train", "infer", "create_datasets", "preprocess"], help="Mode: train, infer, create_datasets, or preprocess")
    parser.add_argument("data_path", help="Path to dataset (for training or preprocessing)", nargs="?")
    parser.add_argument("model_path", help="Path to trained model (for inference)", nargs="?")
    parser.add_argument("input_text", help="Text input for inference", nargs="?")
    parser.add_argument("language", help="Target language for dataset creation (e.g., 'ta' for Tamil, 'te' for Telugu)", nargs="?")

    args = parser.parse_args()

    ensure_dir("./results")

    if args.mode == "train":
        if not args.data_path:
            raise ValueError("data_path is required for training mode")
        train_model(args.data_path)

    elif args.mode == "infer":
        if not args.model_path or not args.input_text:
            raise ValueError("model_path and input_text are required for inference mode")
        infer = SentimentInference(args.model_path)
        sentiment = infer.predict(args.input_text)
        print(f"Predicted Sentiment: {sentiment}")

    elif args.mode == "create_datasets":
        if not args.language or not args.data_path:
            raise ValueError("language and data_path are required for create_datasets mode")
        target_file = f"data/amazon_reviews_{args.language}.csv"
        translate_reviews(
            input_file=args.data_path,
            output_file=target_file,
            target_language=args.language
        )
        print(f"Dataset created for language {args.language} at {target_file}")

    elif args.mode == "preprocess":
        if not args.data_path:
            raise ValueError("data_path is required for preprocess mode")
        train_data, test_data = preprocess_data(args.data_path)
        train_data.to_csv("data/train_data.csv", index=False)
        test_data.to_csv("data/test_data.csv", index=False)
        print("Preprocessed data saved to data/train_data.csv and data/test_data.csv")