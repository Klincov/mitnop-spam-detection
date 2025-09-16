import os
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text: str) -> str:
    """Osnovno čišćenje teksta: mala slova, uklanjanje brojeva, URL-ova i višestrukih razmaka."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # URL-ovi
    text = re.sub(r"\d+", " ", text)             # brojevi
    text = re.sub(r"\s+", " ", text).strip()     # višestruki razmaci
    return text

def prepare_data(filepath, num_words=5000, max_len=100, mode="sequence", tokenizer=None, vectorizer=None):
    """
    mode: "sequence" (za RNN) ili "tfidf" (za FFNN)
    tokenizer / vectorizer: ako je dat, koristi ga umesto ponovnog fitovanja
    """
    # 1. Učitavanje fajla
    data = pd.read_csv(filepath, sep="\t", header=None, encoding="latin-1")
    data.columns = ["label", "text"]

    # 2. Čišćenje teksta
    data["text"] = data["text"].apply(clean_text)

    # 3. Label encoding
    encoder = LabelEncoder()
    data["label"] = encoder.fit_transform(data["label"])
    y = data["label"].values

    # 4. TF-IDF ili sekvence
    if mode == "tfidf":
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=num_words)
            X = vectorizer.fit_transform(data["text"]).toarray()
        else:
            X = vectorizer.transform(data["text"]).toarray()
        return X, y, vectorizer

    elif mode == "sequence":
        if tokenizer is None:
            tokenizer = Tokenizer(num_words=num_words)
            tokenizer.fit_on_texts(data["text"])
        sequences = tokenizer.texts_to_sequences(data["text"])
        X = pad_sequences(sequences, maxlen=max_len)
        return X, y, tokenizer

    else:
        raise ValueError("Nepoznat mode. Dozvoljeni su 'sequence' ili 'tfidf'.")
