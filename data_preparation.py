import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import os
import pickle
import numpy as np

def prepare_data(file_names):
    # Khởi tạo stemmer
    stemmer = LancasterStemmer()

    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Duyệt qua từng tệp intents.json
    for file_name in file_names:
        # Mở và đọc file intents.json
        with open(file_name) as file:
            data = json.load(file)

        # Duyệt qua các intent trong dữ liệu
        for intent in data["intents"]:
            # Duyệt qua các pattern trong mỗi intent
            for pattern in intent["patterns"]:
                # Tách pattern thành các từ
                wrds = nltk.word_tokenize(pattern)
                # Thêm các từ vào danh sách words
                words.extend(wrds)
                # Thêm các từ và tag tương ứng vào docs_x và docs_y
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            # Nếu tag chưa có trong labels, thêm tag vào labels
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    # Stemming và loại bỏ các từ trùng lặp trong words
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    # Sắp xếp labels
    labels = sorted(labels)

    # Khởi tạo training và output
    training = []
    output = []

    # Khởi tạo một danh sách chứa số 0 với độ dài bằng số lượng labels
    out_empty = [0 for _ in range(len(labels))]

    # Duyệt qua docs_x
    for x, doc in enumerate(docs_x):
        bag = []

        # Stemming các từ trong doc
        wrds = [stemmer.stem(w.lower()) for w in doc]

        # Tạo bag of words
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Tạo output cho mỗi câu
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        # Thêm bag và output_row vào training và output
        training.append(bag)
        output.append(output_row)

    # Chuyển đổi training và output thành numpy array
    training = np.array(training)
    output = np.array(output)

    # Lưu words, labels, training, và output vào file data.pickle
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

    # Trả về words, labels, training, và output
    return words, labels, training, output

# Tạo hàm để lấy các file trong thư mục stories
def file_name():
    folder_path = "stories"
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]
    return file_names

file_names_list = file_name()