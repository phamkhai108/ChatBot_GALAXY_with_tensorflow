import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import os
import unidecode 
from model import create_model
from data_preparation import prepare_data, file_names_list
from tom_tat import tom_tat_van_ban
from acronym.stand_words import normalize_text, dictions

# Khởi tạo stemmer
stemmer = LancasterStemmer()

# Chuẩn bị dữ liệu và tạo mô hình
words, labels, training, output = prepare_data(file_names_list)
model = create_model()

# Khởi tạo biến data
data = {"intents": []}

# Lặp qua tất cả các file JSON trong thư mục "stories" và mở các file json trong đó
folder_path = "stories"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path) as file:
            file_data = json.load(file)
            data["intents"].extend(file_data["intents"])

# Biến toàn cục để kiểm tra xem có đang chờ tóm tắt văn bản không
waiting_for_summary = False

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
        # nếu chat khi đang đợi tóm tắt thì gọi lại mô hình
        results = model.predict(np.array([bag_of_words(user_input, words)]))
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        # nếu mô hình dự đoán thuộc về một câu hỏi nào đó
        if results[0, results_index] > 0.95:
            waiting_for_summary = False
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    if responses:
                        bot_response = random.choice(responses)
                        return bot_response
                    else:
                        return "Xin lỗi, tôi không hiểu bạn đang nói gì."

        # thực hiện tóm tắt 
        summary = tom_tat_van_ban(user_input)
        waiting_for_summary = False
        return summary

    inp = user_input.lower()
    inp = normalize_text(inp, dictions)
    inp = unidecode.unidecode(inp)
    results = model.predict(np.array([bag_of_words(inp, words)]))
    results_index = np.argmax(results)
    tag = labels[results_index]
    
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    
    if tag == "tom_tat":
        bot_response = random.choice(responses) if responses else "Xin lỗi, tôi không hiểu bạn đang nói gì."
        waiting_for_summary = True
        return bot_response

    # kiểm tra chỉ số dự đoán bé hơn 70% thì in ra 
    elif results[0, results_index] < 0.95: 
        for tg in data["intents"]:
            if tg['tag'] == "khong_hieu":
                responses = tg['responses']
        bot_response = random.choice(responses) if responses else "Xin lỗi, tôi không hiểu bạn đang nói gì."
        return bot_response
    else: 
        bot_response = random.choice(responses) if responses else "Xin lỗi, tôi không hiểu bạn đang nói gì."
        return bot_response

# Sử dụng bot
# print(chat("tom tat van ban"))
# print(chat(" Ý kiến của tác giả vô cùng đúng đắn, chính xác. Bởi “nghĩa tiêu dùng, nghĩa tự vị” của chữ là những lớp nghĩa chung, được sử dụng trong giao tiếp hằng ngày, bất kì ai cũng hiểu. Vì vậy, người làm thơ phải tạo ra được những con chữ riêng cho bản thân mình. Nhà thơ phải tạo ra được những ngôn ngữ nghệ thuật riêng, gửi gắm được tiếng lòng của bản thân để tạo nên độ vang và sức gợi cảm. Cấu trúc ngôn từ của một bài thơ sẽ làm nên giá trị của bài thơ đó."))

# print("Bắt đầu chat với bot! (chat 'quit' để dừng chatbot)")
# while True:
#     user_input = input("you: ")
#     if user_input.lower() == 'quit':
#         break
#     bot_response = chat(user_input)
#     print("bot:", bot_response)
