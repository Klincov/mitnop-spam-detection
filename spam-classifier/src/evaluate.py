import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from preprocess import prepare_data

# Putanje modela i pomoćnih fajlova
MODEL_FFNN_PATH = "models/spam_ffnn.h5"
MODEL_RNN_PATH = "models/spam_rnn.h5"
VECTORIZER_PATH = "models/vectorizer.pkl"
TOKENIZER_PATH = "models/tokenizer.pkl"

if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "spam")

    # --- 1. Evaluacija FFNN (TF-IDF) ---
    if os.path.exists(MODEL_FFNN_PATH) and os.path.exists(VECTORIZER_PATH):
        print("\n=== Evaluacija FFNN (TF-IDF) ===")

        # Učitaj vectorizer
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        # Pripremi podatke sa istim vectorizerom
        X, y, _ = prepare_data(filepath, num_words=5000, mode="tfidf",vectorizer=vectorizer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Učitaj model
        ffnn = load_model(MODEL_FFNN_PATH)

        # Evaluacija
        loss, acc = ffnn.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}")

        y_pred = (ffnn.predict(X_test) > 0.5).astype("int32")

        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham", "Spam"],
                    yticklabels=["Ham", "Spam"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - FFNN")
        plt.show()
    else:
        print("FFNN ili vectorizer nisu pronađeni. Prvo pokreni fit.py")

    # --- 2. Evaluacija RNN (sekvence) ---
    if os.path.exists(MODEL_RNN_PATH) and os.path.exists(TOKENIZER_PATH):
        print("\n=== Evaluacija RNN (sekvence) ===")

        # Učitaj tokenizer
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        # Pripremi podatke sa istim tokenizerom
        X, y, _ = prepare_data(filepath, num_words=5000, max_len=100, mode="sequence", tokenizer=tokenizer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Učitaj model
        rnn = load_model(MODEL_RNN_PATH)

        # Evaluacija
        loss, acc = rnn.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}")

        y_pred = (rnn.predict(X_test) > 0.5).astype("int32")

        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Ham", "Spam"],
                    yticklabels=["Ham", "Spam"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - RNN")
        plt.show()
    else:
        print("RNN ili tokenizer nisu pronađeni. Prvo pokreni fit.py")
