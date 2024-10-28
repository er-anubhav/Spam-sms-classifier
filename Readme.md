# SMS Spam Classifier

This project is an SMS spam classifier that uses natural language processing (NLP) to identify and filter out spam messages from genuine ones. The model is deployed using [Streamlit](https://streamlit.io/), which allows users to interact with the classifier through a web application.

**Streamlit App**: [SMS Spam Classifier on Streamlit](https://spam-sms-classifier-anubhav.streamlit.app/)

---

## Project Overview

In this project, we:
1. Clean and preprocess the text data by removing stopwords and punctuation.
2. Build and train a machine learning model to classify SMS messages as "spam" or "ham" (not spam).
3. Deploy the model on Streamlit, allowing users to input SMS messages and see the classification results in real-time.

## Dataset

The dataset consists of SMS messages labeled as "spam" or "ham". It includes a variety of texts, from promotional messages to personal communications. The dataset has been preprocessed to remove stopwords and punctuation, ensuring that only meaningful words contribute to the classification.

---

## Setup Instructions

### Prerequisites

Ensure that the following packages are installed:
- `pandas`
- `scikit-learn`
- `nltk`
- `streamlit`

You can install them using:
```bash
pip install pandas scikit-learn nltk streamlit
