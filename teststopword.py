from underthesea import word_tokenize
def stopword(text):
    # Đọc file stopword.txt và tạo danh sách từ dừng
    with open('stopword.txt', 'r') as f:
        stopwords = f.read().splitlines()
    # Tách từ
    tokens = word_tokenize(text, format="text")

    # Loại bỏ từ dừng
    filtered_tokens = [token for token in tokens.split() if token not in stopwords]

    for i in filtered_tokens:
        tokens = filtered_tokens.replace('_', ' ')

    text_last = ' '.join(filtered_tokens)
    return text_last
text = "chàng là một người hiền lành và cũng là một con người tốt bụng"
print(stopword(text))

