import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def prepare_data(filepath, num_words=5000, max_len=100):
    # 1. Učitavanje fajla sa tab delimiterom
    data = pd.read_csv(filepath, sep="\t", header=None, encoding="latin-1")
    data.columns = ["label", "text"]

    # 2. Label encoding (ham=0, spam=1)
    encoder = LabelEncoder()
    data["label"] = encoder.fit_transform(data["label"])

    # 3. Tokenizacija
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data["text"])
    sequences = tokenizer.texts_to_sequences(data["text"])

    # 4. Padding sekvenci
    X = pad_sequences(sequences, maxlen=max_len)
    y = data["label"].values

    return X, y, tokenizer

from sklearn.model_selection import train_test_split

# Poziv funkcije
X, y, tokenizer = prepare_data("spam-classifier/data/spam", num_words=5000, max_len=100)

# Deljenje na trening i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (X,y,tokenizer)
