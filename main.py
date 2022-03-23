import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import json
import pickle
# nltk.download('punkt')

stemmer = LancasterStemmer()

# Load in the json file
with open('intents.json') as file:
    data = json.load(file)

# print(data["intents"][0])

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

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
        wrds = [stemmer.stem(w) for w in doc if w != "?"]
        # print(wrds)
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        # print(bag)
        # Copy the array
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    # saving the variables to "data.pickle" so anytime we run, we don't have to go through the preprocessing.
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output ), f)

network = tflearn.input_data(shape=[None, len(training[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation = "softmax")
network = tflearn.regression(network)

model = tflearn.DNN(network)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch = 1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

