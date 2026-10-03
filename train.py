"""Train the churn classifier and save it to disk."""
from pathlib import Path

from model.classifier import train_model

MODEL_PATH = Path("models/churn_classifier.joblib")


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_model(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
