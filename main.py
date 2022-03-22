import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow
import random
import json

# nltk.download('punkt')

stemmer = LancasterStemmer()

# Load in the json file
with open('intents.json') as file:
    data = json.load(file)

# print(data["intents"][0])

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
# print(words)
print(docs_x)
words = [stemmer.stem(w.lower()) for w in words]

# words = sorted(list(set(words)))
words = list(set(words))
print(words)

labels = sorted(labels)
# print(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    # Copy the array
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

