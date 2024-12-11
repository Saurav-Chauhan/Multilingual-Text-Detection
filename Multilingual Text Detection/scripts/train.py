import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from models.multilingual_model import MultilingualSentimentModel
from scripts.preprocess import preprocess_data

def train_model(data_path, model_name="xlm-roberta-base"):
    train_data, test_data = preprocess_data(data_path)

    # Convert data to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    model = MultilingualSentimentModel(model_name)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.tokenizer,
    )

    trainer.train()
    model.model.save_pretrained("./trained_model")
    model.tokenizer.save_pretrained("./trained_model")