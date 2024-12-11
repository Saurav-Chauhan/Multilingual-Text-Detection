from sklearn.metrics import classification_report

def evaluate_model(predictions, labels):
    report = classification_report(labels, predictions, target_names=["Negative", "Neutral", "Positive"])
    return report