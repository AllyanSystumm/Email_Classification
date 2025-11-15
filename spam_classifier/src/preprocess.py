import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')   # <-- add this line
nltk.download('stopwords')

def preprocess_data(data):
    # Initialize the stemmer
    stemmer = PorterStemmer()
    
    # Tokenize the text
    data['tokenized'] = data['Message'].apply(lambda x: word_tokenize(x.lower()))  # Assuming your column name is 'Message'
    
    # Remove stopwords and stem words
    stop_words = set(stopwords.words('english'))
    data['processed'] = data['tokenized'].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
    
    # Join the processed words back into a string
    data['processed_text'] = data['processed'].apply(lambda x: ' '.join(x))
    
    # Return the processed data
    return data

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('data/mail_data.csv')
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Print the processed data
    print(processed_data[['Message', 'processed_text']].head())


from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(data):
    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Convert the processed text to TF-IDF features
    features = tfidf.fit_transform(data['processed_text']).toarray()
    
    # Print the shape of the feature array
    print("Feature shape:", features.shape)
    
    return features, tfidf

if __name__ == "__main__":
    # Load the data
    data = pd.read_csv('data/mail_data.csv')
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Create features from the processed data
    features, tfidf = create_features(processed_data)
    
    # Print a sample feature vector (first row)
    print(features[0])
