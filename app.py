import streamlit as st
import pickle
import string
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def train_model():
    # Example data - replace with your actual dataset
    messages = [
        "Free lottery winner! You've won $1000000",
        "Meeting at 3pm tomorrow",
        "URGENT: Your account has been suspended",
        "Hi Mom, how are you?",
        # Add more examples...
    ]
    labels = [1, 0, 1, 0]  # 1 for spam, 0 for not spam
    
    # Transform all messages
    transformed_messages = [transform_text(message) for message in messages]
    
    # Create and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(transformed_messages)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X, labels)
    
    # Save the model and vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return tfidf, model

def load_model():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
    except FileNotFoundError:
        st.warning("Model files not found. Training new model...")
        tfidf, model = train_model()
    return tfidf, model

def main():
    st.title("Email/SMS Spam Classifier")
    
    # Load or train the model
    tfidf, model = load_model()
    
    # Get user input
    input_sms = st.text_area("Enter the message")
    
    if st.button('Predict'):
        if input_sms:
            # Preprocess
            transformed_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]
            # Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        else:
            st.warning("Please enter a message to classify")

if __name__ == "__main__":
    main()
