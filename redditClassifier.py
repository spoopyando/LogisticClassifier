import random
import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine as cosDist
import numpy as np
import json

#################################################################################################
"""
Function: 
    Take a string and find out which subreddit that string is likely from.

Description: 
    Open a corpus
    Train a classifier using a preloaded word vectors from google
    Make a prediction where the input string is from

Components:
    Corpus - Jsonlist containing objects
        Objects - 1) "body": a subreddit comment, "subreddit": the name of a subreddit
    Preloaded word vectors - From GoogleNews. A .bin file

IMPORTANT:
    The default corpus (Google news vector) can be downloaded from "https://github.com/mmihaltz/word2vec-GoogleNews-vectors".
    This pre-trained word vector model or your own must be included for this script to be properly used.
"""
#################################################################################################

# Pretrained word vectors from google
model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary = True)

# Empty classifier to be updated as a trained classifier
trainedModel = None

def setModel(Model):
    global model
    Model = model
    return

# Find pretrained word vector if it exists
def getVector(word):
    global model
    if word in model:
        return model[word]
    else:
        return np.zeros(300)

# Accunulate word vector values
def addVector(vector, words):
    for word in words:
        vector += getVector(word)
    return vector

def classifier_train(file):
    # Create list of word vectors and list of subreddit labels
    wordVectors = []
    subredditLabels = []

    # List to contain json data
    jsonData = []

    # Open jsonlist file to process lines of the json file
    with open(file) as jsonFile:
        for lines in jsonFile:
            # Parse entire string of json file as a python object. Creates key-object : subreddit-body
            line = json.loads(lines)
            jsonData.append(line)

    for objects in jsonData:
        comment = objects["body"]
        subreddit = objects["subreddit"]
        comment_vector = np.zeros(300)

        # Tokenize the words of the comment for vector addition
        tokenized_comments = comment.split()
        comment_vector += addVector(comment_vector, tokenized_comments)

        # Get average of word vector values and append to list of word vectors
        comment_vector /= len(tokenized_comments)
        wordVectors.append(comment_vector)
        subredditLabels.append(subreddit)
    
    global trainedModel
    # Set up the classifier "shell"
    classifier = LogisticRegression(max_iter = 2000)
    # Train the classifier
    classifier = classifier.fit(wordVectors, subredditLabels)
    trainedModel = classifier

    return

def classifier_test(sentence):
    word_vector = np.zeros(300)
    tokenized_sentence = sentence.split()
    word_vector += addVector(word_vector, tokenized_sentence)
    word_vector /= len(tokenized_sentence)
    subreddit = trainedModel.predict([word_vector])

    return subreddit

def main():
    print("Setting model..")
    setModel(model)
    print("Model has been set!")

    print("Training the classifier..")
    classifier_train("redditComments_train.jsonlist")
    print("Classifier has been trained!")

    sentenceInput = "I enlisted and it was hell"
    print("Sentence input: ", sentenceInput)
    print("The predicted subreddit of the input sentence", classifier_test(sentenceInput))

main()