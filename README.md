# Chatbot
Link: https://colab.research.google.com/drive/1HC3tbDQB2T-6abGKaSkDiLY_lkuwM4Xw#scrollTo=4_G62KiBUp0E

# Description
In this project, I implement a simple AI chatbot that trains on the json data I provide. There are four layers to this neural network model: 1 input layer with the size of unique words on the data, 2 hidden layers with the size of 8, and 1 output layer with the size of the labels or how many "tag" in the json data I provide. In addition, I preprocess the data using tokenization and new word normalization techniques or stemming algorithm called "LancasterStemmer" in order to train the data and predict based on the user's input. I also add a feature where if the model couldn't predict the response with a 50% confidence or above, the bot would just response with "I don't know." 

Resources that I and my teammates have followed or researched for this project:

https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

https://www.educative.io/edpresso/what-is-wordtokenize-in-python

# Install and Run the Project
This project requires to imported and installed Python libraries: tflearn, numpy, pickle, json, random, nltk.
