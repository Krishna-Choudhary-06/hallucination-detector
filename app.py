from src.data.loader import load_dataset
from src.features.feature_engineering import pair_features
from src.models.train import train_model


def main():
    print("🚀 Loading dataset...")
    df = load_dataset()

    print("⚙️ Training model...")
    model, scaler, feature_keys, acc = train_model(df, pair_features)

    print("🎯 Validation Accuracy:", acc)


if __name__ == "__main__":
    main()