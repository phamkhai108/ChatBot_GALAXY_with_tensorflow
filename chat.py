import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import unidecode 
from model import create_model
from data_preparation import prepare_data
from tom_tat import tom_tat_van_ban
from acronym.stand_words import normalize_text, dictions
# Khởi tạo stemmer
stemmer = LancasterStemmer()

# Chuẩn bị dữ liệu và tạo mô hình
words, labels, training, output = prepare_data()
model = create_model()

# Mở và đọc file intents.json
with open("intents.json") as file:
    data = json.load(file)

# Biến toàn cục để kiểm tra xem có đang chờ tóm tắt văn bản không
waiting_for_summary = False

# Biến toàn cục để lưu trữ lịch sử chat
chat_history = ["", ""]

# Hàm tạo túi từ (bag of words)
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Hàm chat với bot
def chat(user_input):
    global waiting_for_summary
    global chat_history
    if waiting_for_summary:
        summary = tom_tat_van_ban(user_input)
        waiting_for_summary = False
        return summary

    # Cập nhật lịch sử chat
    chat_history.append(user_input)
    if len(chat_history) > 2:
        chat_history.pop(0)
    # Tạo một chuỗi từ lịch sử chat
    inp = " ".join(chat_history)
    inp = normalize_text(inp, dictions)
    inp = unidecode.unidecode(inp)
    results = model.predict(np.array([bag_of_words(inp, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    if tag == "tom_tat":
        bot_response = random.choice(responses)
        waiting_for_summary = True
        return bot_response
    #kiểm tra chỉ só dự đoán bé hơn 50% thì in ra 
    elif results[0, results_index] < 0.7: 
        for tg in data["intents"]:
            if tg['tag'] == "khong_hieu":
                responses = tg['responses']
        bot_response = random.choice(responses)
        return bot_response
    else: 
        bot_response = random.choice(responses)
        return bot_response # inp



# print("Bắt đầu chat với bot! (chat 'quit' để dừng chatbot)")
# while True:
#     user_input = input("you: ")
#     if user_input.lower() == 'quit':
#         break
#     # print("you: ", user_input)
#     # if inp.lower() == "quit:": break
#     bot_response = chat(user_input)
#     #in ra đoạn chat trong 2 lượt mới nhất 
#     # print(a)
#     print("bot:",bot_response)