import pandas as pd
from preprocess import build_tokenizer, texts_to_padded
from model import build_model

df = pd.read_csv('data/train.csv')

X = df['text']
y = df['generated']

token = build_tokenizer(X)
X = texts_to_padded(token, X)

model = build_model(len(token.word_index)+1, 300)

model.fit(X, y, validation_split=0.1, epochs=10)

model.save("model/model.keras")
