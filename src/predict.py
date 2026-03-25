import pickle
import json
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_artifacts():
    model = load_model('model.keras')

    with open('tokenizer.pkl', 'rb') as f:
        token = pickle.load(f)

    with open('config.json') as f:
        params = json.load(f)

    return model, token, params


def predict_text(text):
    model, token, params = load_artifacts()

    seq = token.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=params['max_len'])

    pred = model.predict(seq)
    return pred[0][0]


if __name__ == "__main__":
    text = "This movie is amazing"
    prob = predict_text(text)
    print("Probability:", prob)
    print("Class:", int(prob > 0.5))
