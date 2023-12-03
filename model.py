# Xây dựng và huấn luyện mô hình
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
from data_preparation import prepare_data

words, labels, training, output = prepare_data()

def create_model():
    model = Sequential()
    model.add(Dense(8, input_shape=(len(training[0]),)))
    model.add(Dense(8))
    model.add(Dense(len(output[0]), activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    try:
        model.load_weights("model.tflearn")
    except:
        model.fit(training, output, epochs=1000, batch_size=8)
        model.save_weights("model.tflearn")
    
    return model
