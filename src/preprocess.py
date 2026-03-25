from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_tokenizer(texts, num_words=150000):
    token = Tokenizer(num_words=num_words, oov_token="<OOV>")
    token.fit_on_texts(texts)
    return token

def texts_to_padded(token, texts, max_len=300):
    seq = token.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
