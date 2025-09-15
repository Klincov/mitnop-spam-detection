import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import prepare_data
from model import build_ffnn
from model_rnn import build_rnn

from tensorflow.keras.models import load_model

# Putanje gde čuvamo fajlove
MODEL_FFNN_PATH = "models/spam_ffnn.h5"
MODEL_RNN_PATH = "models/spam_rnn.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

if __name__ == "__main__":
    # Ako modeli već postoje, preskoči treniranje
    if (
        os.path.exists(MODEL_FFNN_PATH)
        and os.path.exists(MODEL_RNN_PATH)
        and os.path.exists(TOKENIZER_PATH)
    ):
        print(" FFNN, RNN i tokenizer već postoje. Preskačem treniranje.")
    else:
        print(" Treniranje modela...")

        # 1. Priprema podataka
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "spam")
        
        X, y, tokenizer = prepare_data(filepath, num_words=5000, max_len=100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. Definiši modele
        ffnn = build_ffnn(input_dim=X_train.shape[1])  
        rnn = build_rnn(vocab_size=5000, embedding_dim=64, max_len=100)

        # 3. Treniraj FFNN
        print("\n Treniranje FFNN...")
        ffnn.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=5,
            batch_size=32,
            verbose=1,
        )

        # 4. Treniraj RNN
        print("\n Treniranje RNN...")
        rnn.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=5,
            batch_size=32,
            verbose=1,
        )

        # 5. Evaluacija oba
        for model, name, path in [
            (ffnn, "FFNN", MODEL_FFNN_PATH),
            (rnn, "RNN", MODEL_RNN_PATH),
        ]:
            print(f"\n=== Evaluacija {name} ===")
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"{name} Test accuracy: {acc:.4f}")

            # Predikcije
            y_pred = (model.predict(X_test) > 0.5).astype("int32")

            print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

            # Konfuziona matrica
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {name}")
            plt.show()

            # Sačuvaj model
            os.makedirs("models", exist_ok=True)
            model.save(path)
            print(f" {name} sačuvan u {path}")

        # 6. Čuvanje tokenizera (zajednički za oba modela)
        with open(TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer sačuvan u {TOKENIZER_PATH}")
