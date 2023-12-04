from underthesea import ner, sent_tokenize
import numpy as np
import pickle
import nltk
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

# Đoạn văn cần tóm tắt
def tom_tat_van_ban(contents):
    # Tiền xử lý và tách câu
    contents_parsed = [content.lower().strip() for content in contents]
    sentences = nltk.sent_tokenize(contents_parsed[0])
    #đêm só câu trong văn bản. Mỗi câu kết thức bỏi dấu .
    # a = len(sentences)
    # print('số câu',a)
    # Sử dụng NER để lấy từ được gắn nhãn
    word_labels = []
    for sentence in sentences:
        sentence_labels = ner(sentence)
        word_labels.extend([label[0] for label in sentence_labels])

    # Lọc các từ không phải là thực thể
    filtered_words = [word for word in word_labels if word not in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']]

    # Biểu diễn vector cho mỗi câu
    X = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        sentence_vec = np.zeros((len(filtered_words)))  # Sử dụng số chiều là độ dài danh sách các từ đã lọc
        for idx, word in enumerate(filtered_words):
            if word in words:
                sentence_vec[idx] = 1  # Đánh dấu 1 nếu từ có trong câu
        X.append(sentence_vec)

    n_clusters = int(a * (35/100))
    if n_clusters >= (a - 1):
        return 'không thể tóm tắt. Yêu cầu câu tóm tắt vượt quá số câu trong văn bản nhập vào!'
    else:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(X)

        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return summary
        # print(summary)

# tom_tat_van_ban(contents)

