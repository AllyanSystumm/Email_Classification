import os
import pickle

def save_model(model, filename):
    # Use the absolute path to the 'models' folder
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)  # Create the folder if it doesn't exist
    
    file_path = os.path.join(models_dir, filename)  # Full path to save the model
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {file_path}")

def load_model(filename):
    models_dir = os.path.join(os.getcwd(), 'models')
    file_path = os.path.join(models_dir, filename)
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)
