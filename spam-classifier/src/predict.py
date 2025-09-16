import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Putanje modela i tokenizera
MODEL_FFNN_PATH = "models/spam_ffnn.h5"
MODEL_RNN_PATH = "models/spam_rnn.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

MAX_LEN = 100  # isto kao kod treniranja

def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("Tokenizer nije pronađen. Prvo istreniraj model.")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

def predict_message(message, model_type="rnn"):
    """
    model_type: "ffnn" ili "rnn"
    """
    tokenizer = load_tokenizer()

    # Učitaj odgovarajući model
    if model_type.lower() == "ffnn":
        model_path = MODEL_FFNN_PATH
    elif model_type.lower() == "rnn":
        model_path = MODEL_RNN_PATH
    else:
        raise ValueError("model_type mora biti 'ffnn' ili 'rnn'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_type.upper()} nije pronađen.")

    model = load_model(model_path)

    # Tokenizacija
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Predikcija
    pred = model.predict(padded)[0][0]
    label = "SPAM" if pred > 0.5 else "HAM"

    return label, float(pred)

if __name__ == "__main__":
    # Primer: unosiš poruku
    poruka = input("Unesi poruku za klasifikaciju: ")

    # Izbor modela
    izbor = input("Koji model želiš koristiti? (ffnn/rnn): ").strip().lower()

    label, score = predict_message(poruka, model_type=izbor)
    print(f"\nRezultat ({izbor.upper()}): {label} ({score:.4f})")
