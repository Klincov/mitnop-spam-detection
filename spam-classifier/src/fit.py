import os
import pickle
from sklearn.model_selection import train_test_split
from preprocess import prepare_data
from model import build_ffnn
from model_rnn import build_rnn

from tensorflow.keras.models import load_model

# Putanje gde čuvamo fajlove
MODEL_PATH = "models/spam_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

if __name__ == "__main__":
    # Ako model već postoji, preskoči treniranje
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
        print("✅ Model i tokenizer već postoje. Preskačem treniranje.")
    else:
        print("Treniranje modela...")

        # 1. Priprema podataka
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "spam")

        X, y, tokenizer = prepare_data(filepath, num_words=5000, max_len=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Inicijalizacija modela
        #model = build_ffnn(X_train.shape[1])  
        # Ako želiš RNN umesto FFNN:
        model = build_rnn(vocab_size=5000, embedding_dim=64, max_len=100)

        # 3. Treniranje
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=5,
            batch_size=32,
            verbose=1
        )

        # 4. Evaluacija
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}")

        # 5. Čuvanje modela
        os.makedirs("models", exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model sačuvan u {MODEL_PATH}")

        # 6. Čuvanje tokenizera
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer sačuvan u {TOKENIZER_PATH}")
