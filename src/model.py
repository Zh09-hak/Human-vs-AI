from keras.layers import Input, Embedding, Conv1D
from keras.layers import GlobalMaxPooling1D, Dense, Dropout
from keras.layers import Bidirectional, LSTM
from keras.models import Model

def build_model(vocab_size, max_len):
    input_layer = Input(shape=(max_len,))

    x = Embedding(vocab_size, 128)(input_layer)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
