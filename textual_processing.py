import RL
import os
import cProfile
# Data pre-processing
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in RL.intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        word = RL.nltk.word_tokenize(pattern)
        words.extend(word)
        
        # adding document in the corpus
        documents.append((word, intent['tag']))

        # add the formula tag to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(documents)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Now, we're looking for reduce all the canonical words – so we can reduce the number of total words in our vocabulary.
# Lemmatization, is reducing duplicates words that are similar to each other. (e.g. play, playing, plays, played)

## lemmatize and lower each word and remove duplicates
words = [RL.lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

## sort classes
classes = sorted(tuple(set(classes)))

## document = combination between patterns and intents
print(len(documents), "documents")

## classes = intents
print(len(classes), "classes", classes)

## words = all words, vocabulary
print(len(words), "unique lemmatize words", words)


RL.pickle.dump(words, open(os.path.abspath('words.pkl'), 'wb'))
RL.pickle.dump(classes, open(os.path.abspath('classes.pkl'), 'wb'))