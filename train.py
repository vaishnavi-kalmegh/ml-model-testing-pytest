from pathlib import Path
from model.classifier import train_model
if __name__=="__main__":
    path=Path("models/churn_classifier.joblib")
    train_model(path)
    print(f"Model trained and saved to {path}")
