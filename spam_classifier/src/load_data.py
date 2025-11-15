import pandas as pd

def load_data():
    # Load the dataset
    data = pd.read_csv('data/mail_data.csv')  # Use relative path directly from the current directory

    
    # Print the first few rows to verify
    print(data.head())
    
    return data

if __name__ == "__main__":
    load_data()
