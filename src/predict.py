import pickle
import json
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_text(text):
    model = load_model("model/model.keras")

    with open("model/tokenizer.pkl", "rb") as f:
        token = pickle.load(f)

    with open("model/config.json") as f:
        params = json.load(f)

    seq = token.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=params["max_len"])

    pred = model.predict(seq)
    return pred[0][0]
