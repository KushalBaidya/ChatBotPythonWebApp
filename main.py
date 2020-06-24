import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import os

import numpy
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
import json
import pickle


with open("intents.json") as file:
    data = json.load(file)


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

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


training = numpy.array(training)
output = numpy.array(output)

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(len(training[0]),)))

model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(len(output[0]), activation="softmax"))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


try:
    model = tf.keras.models.load_model("chatbot.h5")
except:
    history = model.fit(training,output, epochs=1000, batch_size=20)
    model.save("chatbot.h5")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat(inp):
    print("Start talking with the bot!")
    results = model.predict(bag_of_words(inp, words).reshape(1,len(words)))[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[results_index]>0.50:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return(random.choice(responses))
    else:
        return("Sorry I didn't get you")

