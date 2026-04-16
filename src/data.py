"""
Quick exploratory data analysis script.

Run directly to inspect the validation split:
    python data.py
"""

from data_helper import load_data, visualize_data, separate_features_target
from config import VAL_CSV


def main():
    df = load_data(VAL_CSV)

    print("Shape:", df.shape)
    print(df.head())
    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isnull().sum().sum())
    print("Class distribution:\n", df["Class"].value_counts())

    visualize_data(df)


if __name__ == "__main__":
    main()