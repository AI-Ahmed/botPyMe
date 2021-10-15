import os
import random
import nltk
from nltk.stem import WordNetLemmatizer

import pickle
import json
import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create an instance object from constructor WordNetLemmatizer to reduce number of canonical words
lemmatize = WordNetLemmatizer()

# Let's load our words & classes data beside our intents json file, and our model

intents = json.loads(open(os.path.abspath('intents.json')).read())
words = pickle.load(open(os.path.abspath("words.pkl"), 'rb'))
classes = pickle.load(open(os.path.abspath("classes.pkl"), 'rb'))

model = load_model("chatPyMe.h5")

def clean_sentences(sentence: str)->list:
    # tokenize the patterns - splitting words into an array
    words_sentence = nltk.word_tokenize(sentence)
    
    # stemming every words - reduce the canonical words
    words_sentence = [lemmatize.lemmatize(word.lower()) for word in words_sentence]
    
    return words_sentence


def bag_of_words(sentence: str, words: list, show_details:bool=True)->np.array:
    # tokenize and lemmatize the patterns
    words_sentences = clean_sentences(sentence)
    
    # create a zero matrix to append list of vocabulary
    bag = np.zeros(len(words))
    
    # Now, we need to check if the user words of sentence found in our words dataset
    for word_in_sentence in words_sentences:
        for index, word in enumerate(words):
            if word == word_in_sentence:
                # assign (1) if current word is in the vocabulary position
                bag[index] = 1
                if show_details:
                    print(f"found in bag: {word}")
    return bag.astype(int)


def predict_answer(sentence:str)->list:
    # filter below threshold prediction
    filter_pred = bag_of_words(sentence, words, show_details=False).reshape(1,-1)
    res = model.predict(filter_pred)[0]
    ERROR_THRESHOLD = 0.25
    results = [[index, intent] for index, intent in enumerate(res) if intent > ERROR_THRESHOLD]
    
    # sorting strength probability (sort by the entity of the result)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for i in results:
        return_list.append({"intent": classes[i[0]], 'probability': str(i[1])})
    
    return return_list
    
    
def answer(intent:list, intent_json:json) ->str:
    # get the tag name from the prediction of the user input
    tag = intent[0]['intent']
    
    # get the list of intents from our json file – response with the right answer from the json file
    list_of_intents = intents['intents']
    
    for intent in list_of_intents:
        # if the tag we predicted as the tag found in intents json file, we're going to select one of the provided answers of 'response' key.
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break            
    return result

