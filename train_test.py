import RL
from textual_processing import classes, words, documents
# Create the training data
training = []

# Create empty array for the output
output_zeros = RL.np.zeros(len(classes)).astype(int)

# training set, bag of words for everysentence
for doc in documents:
    # initialize bag of words
    bag = []

    # list of tokenized words for the patterns
    word_patterns = doc[0]

    # lemmatize each word – create base word, in attempt to represent related words
    word_patterns = [RL.lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # append in the bag of words array with 1, if the word is found in the current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # output is a '0' for each tag, and '1' for current tag (for each pattern)
    output_row = list(output_zeros)   # we assigned the empty matrix to new variables to avoid the accumulation that caused by using global variable in the loop. 

    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
   
# shuffle the features and make numpy array
RL.shuffle(training)
training = RL.np.asarray(training)
print(training.shape)
# create training and testing set. X – (a.k.a patterns, entities, words), Y – (a.k.a intents, classes)
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data is created!")
