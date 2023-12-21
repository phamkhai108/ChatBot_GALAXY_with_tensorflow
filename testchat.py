
from underthesea import ner, pos_tag, word_tokenize
text = 'Thời tiết hôm nay ra sao'

with open('stopword.txt', 'r') as f:
        stopwords = f.read().splitlines()
tokens = word_tokenize(text, format="text")
filtered_tokens = [token for token in tokens.split() if token not in stopwords]
tokens = tokens.replace('_', ' ')
# tokens = ' '.join(tokens)
print(tokens)
print(ner(tokens))
# print(filtered_tokens)


