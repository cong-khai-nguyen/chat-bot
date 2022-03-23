# Chatbot


# Description
In this project, I implement a simple AI chatbot that trains on the json data I provide. There are four layers to this neural network model: 1 input layer with the size of unique words on the data, 2 hidden layers with the size of 8, and 1 output layer with the size of the labels or how many "tag" in the json data I provide. In addition, I use new techniques called "Word Normalization" and stemming algorithm called "LancasterStemmer" to train the data and predict the user's input. I also add a feature where if the model couldn't predict the response with a 50% confidence or above, the bot would just response with "I don't know." 

Resources that I have followed or researched for this project:
https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
https://www.educative.io/edpresso/what-is-wordtokenize-in-python

# Install and Run the Project
This project requires to imported and installed Python libraries: tflearn, numpy, pickle, json, random, nltk.
