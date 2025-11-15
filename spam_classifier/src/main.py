from load_data import load_data
from preprocess import preprocess_data, create_features
from models import train_models
from evaluate import save_model, load_model
import pickle

def main():
    # Load data
    data = load_data()
    
    # Preprocess data
    processed_data = preprocess_data(data)
    
    # Create features (TF-IDF)
    features, tfidf = create_features(processed_data)
    
    # Get the labels (spam = 1, ham = 0)
    labels = processed_data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    # Train models
    nb_model, logreg_model = train_models(features, labels)

    # Save the trained models
    save_model(nb_model, '../models/naive_bayes_model.pkl')
    save_model(logreg_model, '../models/logreg_model.pkl')

    # Save the TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
     pickle.dump(tfidf, f)

if __name__ == "__main__":
    main()
