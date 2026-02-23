from src.data_loader import load_data
from src.data_cleaning import clean_data
from src.data_transformation import transform_data
from src.train_model import train_models

def main():

    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Transforming data...")
    df = transform_data(df)

    print("Training models...")
    train_models(df)

if __name__ == "__main__":
    main()