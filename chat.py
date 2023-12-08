# Hàm chat với bot
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from data_preparation import prepare_data
from model import create_model
from tom_tat import tom_tat_van_ban
#sử dùng hàm 
from acronym.stand_words import normalize_text, dictions

# Khởi tạo stemmer
stemmer = LancasterStemmer()

# Chuẩn bị dữ liệu và tạo mô hình
words, labels, training, output = prepare_data()
model = create_model()

# Mở và đọc file intents.json
with open("intents.json") as file:
    data = json.load(file)

def bag_of_words(s, words):
    # Khởi tạo bag of words với số lượng từ bằng số lượng words
    bag = [0 for _ in range(len(words))]

    # Tách câu thành các từ và stem các từ
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # Nếu từ có trong câu, đánh dấu 1 trong bag of words
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    # Trả về bag of words dưới dạng numpy array
    return np.array(bag)

def chat():
    # In ra thông báo bắt đầu chat
    print("Bắt đầu chat với bot! (chat 'quit' để dừng chatbot)")
    chat_history = ["", ""]
    while True:
        # Nhận input từ người dùng, sử dụng hàm normalize_text để giải mã chữ viết tắt
        inp = normalize_text(input("You: "),dictions)
        # print(inp)
        # Nếu người dùng nhập "quit", kết thúc vòng lặp
        if inp.lower() == "quit":
            break
        #cập nhật lịch sử cho câu chat
        chat_history.append(inp)
        if len(chat_history) > 2:
            chat_history.pop(0)
        # Tạo một chuỗi từ lịch sử chat
        chat = " ".join(chat_history)
        # Dự đoán tag của câu người dùng nhập
        results = model.predict(np.array([bag_of_words(chat, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]

        # Tìm phản hồi tương ứng với tag
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        # Nếu tag là "tom_tat", yêu cầu người dùng nhập đoạn văn cần tóm tắt
        if tag == "tom_tat":
            print(random.choice(responses))
            content = input("You: ")
            summary = tom_tat_van_ban(content)
            print(chat)
            print(summary)
        else:
            print(chat)
            print(random.choice(responses))
chat()
