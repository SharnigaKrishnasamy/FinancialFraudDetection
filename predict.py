import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from model.hdc_encoding import encode_sample

model = Sequential([
    Dense(64, activation='relu', input_shape=(1000,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

def predict_transaction(data):
    hv = encode_sample(data)
    hv = hv.reshape(1, -1)

    prob = model.predict(hv)[0][0]
    result = "🚨 Fraud Transaction" if prob > 0.5 else "✅ Safe Transaction"

    return prob, result