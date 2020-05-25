######################################
# word2Vec.py                        #
#                                    #
# V 1.0                              #
# Class embedding the word           #
# embedding process using Word2Vec   #
# model from Gensim Library.         #
#                                    #
# Author: Thomas Di Martino          #
#                                    #
# License: Apache License V2.0       #
######################################

import os
import sys
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Used to import libraries from an absolute path starting with the project's root
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import multiprocessing

# Adding product_linker to path to have absolute import enabled: crucial to have coherent file architecture when executing jupyter notebooks
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils.utils import _debug
from src.exceptions.modelException import UntrainedModelException

class Word2VecModel():

    def __init__(self, vector_size=50, min_count=20, workers=multiprocessing.cpu_count()-1, debug=False, lang="english", detect_bigrams=False, epochs = 30):
        self._vector_size = vector_size
        """Parameterizes the size of the word vectors"""
        self._min_count = min_count
        """ Minimum count of appearance of each word to be added to the vocabulary. The smallest, the more words will be considered known but the hardest the training will be. """
        self._workers = workers
        """ Number of workers used for computation. By default: cpu count - 1 """
        self._is_debug = debug
        """ Debug variable. If true, activates debug mode """
        self._stop_words = None
        """ Lazy loading of a stop words vocabulary """
        self._lang = lang
        """ Language of the input vocabulary """
        self._bigram_model = BigramModel(debug=debug)
        """ Placeholder for the bigram model eventually used when training """
        self._detect_bigrams = detect_bigrams
        """ Parameterizes if diagrams are detected or not """
        self._epochs = epochs
        """ Number of epochs to train the model """
        self._fitted = False
        """ variable that checks if the model has been fitted to raise an error if transform is called without training """

    def _cleaning(self, doc):
        """
            Removes stop word from a list of words
        """
        if (self._stop_words == None):
            self._stop_words = set(stopwords.words(self._lang))

        return [
            wordFiltered for wordFiltered in word_tokenize(doc) if wordFiltered not in self._stop_words
        ]

    def _tokenize_and_filter(self, X):
        """
            Tokenizes a sentence and call the _cleaning method on the resulting tokenized title
        """
        _debug(self._is_debug, "Step 1: Tokenizing and filtering titles...")

        return [
            self._cleaning(text) for text in tqdm(X)
        ]

    def fit(self, X):
        """
        Fits the word2Vec model to the input data.
        If bigram are to be detected, also fits the embedded bigram model.
        """

        _debug(self._is_debug, "\tTitles tokenized and filtered.")

        tokenized_and_filtered = self._tokenize_and_filter(X)

        if (self._detect_bigrams):
            
            #Step 2. Detect Bigrams
            _debug(self._is_debug, "Step 1.b: Detect bigrams")


            _debug(self._is_debug, "\tTraining bigram model")
            
            tokenized_and_filtered = self._bigram_model.fit_transform(tokenized_and_filtered)
        
            _debug(self._is_debug, "\tBigrams computed.")

        _debug(self._is_debug, "Step 2: Initialising Word2Vec model..")
        
        # Step 3. Encode the words
        self._w2v_model = Word2Vec(min_count=self._min_count, size=self._vector_size, workers=self._workers)
        _debug(self._is_debug, "\tBuilding Word2Vec vocabulary")
    
        self._w2v_model.build_vocab(tokenized_and_filtered)

        _debug(self._is_debug, "\tTraining Word2Vec model")

        self._w2v_model.train(tokenized_and_filtered, total_examples=self._w2v_model.corpus_count, epochs=self._epochs)
        _debug(self._is_debug, "\tWord2Vec model trained")
        self._w2v_model.init_sims(replace=True)

        self._fitted = True

        return self

    def transform(self, X):
        """
        Transforms the input data using the trained word2vec model.
        """
        if (not self._fitted):
            raise UntrainedModelException()

        tokenized_and_filtered = self._tokenize_and_filter(X)

        if (self._detect_bigrams):
            tokenized_and_filtered = self._bigram_model.transform(tokenized_and_filtered)
        
        _debug(self._is_debug, "Step 3: Encoding the words on the newly trained word2vec model")

        res = []
        count_unknown_word = 0
        for sentence in tqdm(tokenized_and_filtered):
            encoded_sentence = []
            for word in sentence:
                vec = None
                try:
                    vec = self._w2v_model[word]
                except:
                    count_unknown_word += 1
                    #_debug(self._is_debug, f"Word {word} passed in sentence {i}, unknown..", msg_type="WARN")
                if vec is not None:
                    encoded_sentence.append(vec)

            res.append(encoded_sentence)

        _debug(self._is_debug, f"Total of {count_unknown_word} unknown_words", msg_type="WARN")
        return res

    def fit_transform(self, X):
        """
        Iteratively calls the fit and transform methods of the word2vec model on the input data.
        """
        return self.fit(X).transform(X)

class BigramModel():

    def __init__(self, debug = False, min_count = 1, threshold = 1):
        self._min_count = min_count
        """ Variable count the minimum appearance frequency of bigrams """
        self._threshold = threshold
        """ Variable specifying the thresholding parameter of the bigram model """
        self._bigram_model = None
        """ Placeholder variable for the bigram model """
        self._fitted = False
        """ Boolean variable specifying if the model has been trained """

    def fit(self, X):
        """ 
        Fits the bigram model to the input list of sentences. 
        Input has to be a list of sentences where each sentences is a list of string (each string being a word).
        """
        phrases = Phrases(X, min_count=self._min_count, threshold=self._min_count)
        self._bigram_model = Phraser(phrases)
        self._fitted = True
        return self
    
    def transform(self, X):
        """
        Transforms the input list of sentences.
        Input has to be a list of sentences where each sentences is a list of string (each string being a word).
        """
        if not self._fitted:
            raise UntrainedModelException()

        return [
            self._bigram_model[sentence] for sentence in X
        ]

    def fit_transform(self, X):
        """
        Fits the model on the input data and updates the input data with the trained model.
        """
        return self.fit(X).transform(X)