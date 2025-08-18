import pandas as pd
import os

def load_and_prepare_data(csv_path: str):
    """
    Load IMDb dataset from CSV, clean it, and return a DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure necessary columns exist
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_columns}")

    # Remove duplicates and missing values
    df.drop_duplicates(subset="review", inplace=True)
    df.dropna(subset=["review", "sentiment"], inplace=True)

    # Map text labels to numeric for model
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == "__main__":
    dataset_path = "data/IMDB Dataset.csv"
    data = load_and_prepare_data(dataset_path)
    print("✅ Data Loaded Successfully!")
    print("Total Samples: ",data.shape)
    print(data.head())
