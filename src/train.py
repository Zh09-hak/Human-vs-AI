import pandas as pd
import pickle
import json

from preprocess import build_tokenizer, texts_to_padded
from model import build_model


def main():
    df = pd.read_csv('data/train.csv')

    X = df['text']
    y = df['generated']

    token = build_tokenizer(X)
    X = texts_to_padded(token, X)

    vocab_size = len(token.word_index) + 1
    max_len = 300

    model = build_model(vocab_size, max_len)

    model.fit(X, y, validation_split=0.1, epochs=5, batch_size=128)

    model.save('model.keras')

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(token, f)

    with open('config.json', 'w') as f:
        json.dump({"max_len": max_len}, f)


if __name__ == "__main__":
    main()
