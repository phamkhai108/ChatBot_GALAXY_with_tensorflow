from acronym.dictions import dictions

# Hàm chuẩn hóa văn bản
def normalize_text(text, dictions):
    words = text.lower().split()
    for i in range(len(words)):
        if words[i] in dictions:
            words[i] = dictions[words[i]]
    a = ' '.join(words)
    return a 

