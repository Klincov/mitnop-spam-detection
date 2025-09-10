from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from preprocess import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def build_rnn(vocab_size=5000, embedding_dim=64, max_len=100):
    model = Sequential()
    # embedding pretvara reči u vektore (npr. 64 dimenzije)
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    # LSTM sloj
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # Priprema podataka
    X, y, tokenizer = prepare_data("data/spam", num_words=5000, max_len=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vocab_size = 5000
    max_len = 100

    model = build_rnn(vocab_size=vocab_size, embedding_dim=64, max_len=max_len)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
