import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Đọc dữ liệu
with open('/home/huythang/PycharmProjects/pythonProject/data/intents.json') as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']

# Tokenize và lemmatize dữ liệu
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Chuyển đổi sang numpy array
training = np.array(training, dtype=object)

# Tách thành X (dữ liệu đầu vào) và y (dữ liệu đầu ra)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Lưu dữ liệu đã chuẩn bị
with open('training_data.pickle', 'wb') as f:
    pickle.dump((words, classes, train_x, train_y), f)

print("Dữ liệu đã được chuẩn bị và lưu vào 'training_data.pickle'")

#File 'training_data.pickle' được sử dụng để lưu trữ dữ liệu đã được chuẩn bị cho việc huấn luyện mô hình chatbot.
# Đây là một cách để lưu trữ và truyền dữ liệu giữa các giai đoạn khác nhau của quá trình xây dựng chatbot