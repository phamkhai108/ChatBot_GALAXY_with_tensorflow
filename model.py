# Xây dựng và huấn luyện mô hình
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle
from data_preparation import prepare_data

# Chuẩn bị dữ liệu
words, labels, training, output = prepare_data()

def create_model():
    # Khởi tạo mô hình Sequential
    model = Sequential()
    # Thêm lớp Dense với 8 units và shape đầu vào là độ dài của training[0]
    model.add(Dense(8, input_shape=(len(training[0]),)))
    # Thêm một lớp Dense khác với 8 units
    model.add(Dense(8))
    # Thêm lớp Dense cuối cùng với số units bằng độ dài của output[0] và hàm kích hoạt là softmax
    model.add(Dense(len(output[0]), activation="softmax"))

    # Biên dịch mô hình với hàm mất mát là categorical_crossentropy, optimizer là adam và metrics là accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    try:
        # Nếu file model.tflearn tồn tại, tải trọng số từ file
        model.load_weights("model.tflearn")
    except:
        # Nếu file model.tflearn không tồn tại, huấn luyện mô hình với dữ liệu training và output
        model.fit(training, output, epochs=1000, batch_size=8)
        # Lưu trọng số của mô hình vào file model.tflearn
        model.save_weights("model.tflearn")
    
    # Trả về mô hình
    return model
