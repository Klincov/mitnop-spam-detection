import os
import pickle
from sklearn.model_selection import train_test_split
from preprocess import prepare_data
from model import build_ffnn
from model_rnn import build_rnn

MODEL_FFNN_PATH = "models/spam_ffnn.h5"
MODEL_RNN_PATH = "models/spam_rnn.h5"
VECTORIZER_PATH = "models/vectorizer.pkl"  # za TF-IDF
TOKENIZER_PATH = "models/tokenizer.pkl"    # za sekvence

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # --- 1. Priprema podataka ---
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "spam")

    # TF-IDF podaci za FFNN
    X_tfidf, y_tfidf, vectorizer = prepare_data(filepath, num_words=5000, mode="tfidf")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_tfidf, y_tfidf, test_size=0.2, random_state=42
    )

    # Sekvence za RNN
    X_seq, y_seq, tokenizer = prepare_data(filepath, num_words=5000, max_len=100, mode="sequence")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # --- 2. FFNN ---
    print("\nTreniranje FFNN (TF-IDF)...")
    ffnn = build_ffnn(input_dim=X_train_f.shape[1])
    ffnn.fit(
        X_train_f, y_train_f,
        validation_data=(X_test_f, y_test_f),
        epochs=5, batch_size=32, verbose=1
    )
    ffnn.save(MODEL_FFNN_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"FFNN i vectorizer sačuvani.")

    # --- 3. RNN ---
    print("\nTreniranje RNN (sekvence)...")
    rnn = build_rnn(vocab_size=5000, embedding_dim=64, max_len=100)
    rnn.fit(
        X_train_r, y_train_r,
        validation_data=(X_test_r, y_test_r),
        epochs=5, batch_size=32, verbose=1
    )
    rnn.save(MODEL_RNN_PATH)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"RNN i tokenizer sačuvani.")
