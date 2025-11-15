# import streamlit as st
# import pickle
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB

# # Load the saved models and vectorizer
# with open('models/naive_bayes_model.pkl', 'rb') as f:
#     nb_model = pickle.load(f)

# with open('models/logreg_model.pkl', 'rb') as f:
#     logreg_model = pickle.load(f)

# with open('models/tfidf_vectorizer.pkl', 'rb') as f:
#     tfidf_vectorizer = pickle.load(f)


# # Function to predict using the models
# def predict_spam_or_not_spam(text, model, tfidf_vectorizer):
#     # Transform the input text using the TF-IDF vectorizer
#     text_tfidf = tfidf_vectorizer.transform([text])
#     # Predict the category (0 = not spam, 1 = spam)
#     prediction = model.predict(text_tfidf)
#     return "Spam" if prediction == 1 else "Not Spam"

# # Streamlit UI
# st.title("Spam vs Not Spam Email Classifier")
# st.write("Enter your email content below:")

# # Text input for email
# email_text = st.text_area("Email Content", "")

# # Button to make prediction
# if st.button("Classify"):
#     if email_text:
#         # Predict using Naive Bayes
#         nb_result = predict_spam_or_not_spam(email_text, nb_model, tfidf_vectorizer)
#         st.write(f"Naive Bayes Prediction: {nb_result}")
        
#         # Predict using Logistic Regression
#         logreg_result = predict_spam_or_not_spam(email_text, logreg_model, tfidf_vectorizer)
#         st.write(f"Logistic Regression Prediction: {logreg_result}")
#     else:
#         st.write("Please enter the email content.")
import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Load the saved models and vectorizer
with open('models/naive_bayes_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('models/logreg_model.pkl', 'rb') as f:
    logreg_model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)


# Function to predict using the models
def predict_spam_or_not_spam(text, model, tfidf_vectorizer):
    # Transform the input text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([text])
    # Predict the category (0 = not spam, 1 = spam)
    prediction = model.predict(text_tfidf)
    return "Spam" if prediction == 1 else "Not Spam"

# Custom CSS for a better look
st.markdown("""
    <style>
    .title {
        color: #FF6347;
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .header {
        color: #4CAF50;
        font-size: 25px;
    }
    .prediction {
        color: #1E90FF;
        font-size: 20px;
        font-weight: bold;
    }
    .btn {
        background-color: #008CBA;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        font-size: 16px;
        border-radius: 5px;
    }
    .btn:hover {
        background-color: #005f73;
    }
    .textarea {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h1 class="title">Spam vs Not Spam Email Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="header">Enter your email content below:</p>', unsafe_allow_html=True)

# Text input for email
email_text = st.text_area("Email Content", "", height=200, max_chars=1000, key="email_text", help="Paste the email content here.", label_visibility="visible")

# Button to make prediction
if st.button("Classify", key="classify_button", help="Click to classify the email", use_container_width=True):
    if email_text:
        # Predict using Naive Bayes
        nb_result = predict_spam_or_not_spam(email_text, nb_model, tfidf_vectorizer)
        st.markdown(f"<p class='prediction'>Naive Bayes Prediction: {nb_result}</p>", unsafe_allow_html=True)
        
        # Predict using Logistic Regression
        logreg_result = predict_spam_or_not_spam(email_text, logreg_model, tfidf_vectorizer)
        st.markdown(f"<p class='prediction'>Logistic Regression Prediction: {logreg_result}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='prediction'>Please enter the email content.</p>", unsafe_allow_html=True)
