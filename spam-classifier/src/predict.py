import os
import pickle
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

MODEL_FFNN_PATH = "models/spam_ffnn.h5"
MODEL_RNN_PATH = "models/spam_rnn.h5"
VECTORIZER_PATH = "models/vectorizer.pkl"
TOKENIZER_PATH = "models/tokenizer.pkl"

def predict_ffnn(text: str):
    # učitaj model i vectorizer
    model = load_model(MODEL_FFNN_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # obradi tekst i napravi tf-idf vektor
    text_clean = clean_text(text)
    X = vectorizer.transform([text_clean])

    # predikcija
    pred = model.predict(X)[0][0]
    return "spam" if pred >= 0.5 else "ham"

def predict_rnn(text: str, max_len=100):

    # učitaj model i tokenizer
    model = load_model(MODEL_RNN_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # obradi tekst i napravi sekvencu
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    X = pad_sequences(seq, maxlen=max_len)

    # predikcija
    pred = model.predict(X)[0][0]
    return "spam" if pred >= 0.5 else "ham"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py [ffnn|rnn] 'Your SMS text here'")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    text = sys.argv[2]

    if model_type == "ffnn":
        result = predict_ffnn(text)
    elif model_type == "rnn":
        result = predict_rnn(text)
    else:
        print("Unknown model type. Use 'ffnn' or 'rnn'.")
        sys.exit(1)

    print(f"Predikcija: {result}")
