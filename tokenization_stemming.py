import nltk
import numpy as np
#print("numpy.__version__", numpy.__version__)

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

#nltk.download('punkt')
object_stemmed_word = PorterStemmer()

def class_tokenize(sentence):
    return nltk.word_tokenize(sentence) #word_tokenize is a function in nltk to break sentence into words

def class_stemming(word):
    word_lower = word.lower()
    return object_stemmed_word.stem(word_lower) #rovided by porterstemmer, to stem off words

def class_bag_of_words(tokenized_sentence, array_of_all_Words): #tokenized senetence. if the tokenized word is available in the sentence, it returns 1
    tokenized_sentence = [class_stemming(w) for w in tokenized_sentence]
    bag = np.zeros(len(array_of_all_Words), dtype=np.float32) #create arrray filled with 0
    for idx, w in enumerate(array_of_all_Words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
        
    return bag #array

# sentence = ["hello"]
# words = ["hello"]
# bag = class_bag_of_words(sentence, words)
# print(bag)

#to check tokenizing
# a = "How long does delivery take?"
# print(a)

# a = class_tokenize(a)
# print(a)

#to check stemming
# words = ["organize", "organ"]
# stemmed_word = [class_stemming(w) for w in words]
# print(stemmed_word)