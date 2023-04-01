'''This modules contains the baseplate methods to:
    - load train, test and validation data.
    - vectorize and embedd data
'''

#--------------- Data Processing ---------------

import pandas as pd
import numpy as np

def _prepare_data(dataframe: pd.DataFrame, split):
    '''Split the dataframe in inputs and labels.
    The labels are divided in two columns, corresponding to each class
    '''
    dataframe['appro'] = 1 - dataframe["class"]
    dataframe['inappro'] = dataframe["class"]

    # dataframe['text'].replace('', np.nan, inplace=True)
    # dataframe.dropna(subset=['text'], inplace=True) 

    return (np.array(dataframe["text"]), np.array(dataframe[["appro", "inappro"]]) if split else np.array(dataframe["class"]))


def load_split_data(split = True):
    '''Load the full set of data splitted into train, test and validation
    '''
    X_train, y_train = _prepare_data(pd.read_csv("data/train.csv"), split)
    X_validate, y_validate = _prepare_data(pd.read_csv("data/validate.csv"), split)
    X_test, y_test = _prepare_data(pd.read_csv("data/test.csv"), split)
    return (X_train, y_train, X_validate, y_validate, X_test, y_test)


def get_text_data():
    return pd.read_csv("data/full.csv")["text"]


#---------- Vectorizing and Embedding ----------
from sklearn.feature_extraction.text import CountVectorizer

def get_split_count_vectorizer(dict=None):
    '''Get split data, if necessary make count vectorizer dictionary.
    Make vectors from data.
    '''
    tockenizer = CountVectorizer(analyzer="word")
    tockenizer.fit(get_text_data())

    X_train, y_train, X_validate, y_validate, X_test, y_test = load_split_data()

    X_train = tockenizer.transform(X_train)
    X_validate = tockenizer.transform(X_validate)
    X_test = tockenizer.transform(X_test)

    return X_train, y_train, X_validate, y_validate, X_test, y_test

import os
import requests
import zipfile

def get_glove_model():
    '''Dowload pretrained glove model (1.7GB), and set up folder architecture'''

    if not os.path.exists("glove"):
        os.makedirs("glove")
    if not os.path.exists("pretrained\glove"):
        os.makedirs("pretrained\glove")

    open('pretrained/glove/glove.twitter.27B.zip', 'wb').write(
    requests.get('https://nlp.stanford.edu/data/glove.twitter.27B.zip', allow_redirects=True).content)
    with zipfile.ZipFile('pretrained/glove/glove.twitter.27B.zip', 'r') as zip_ref:
        zip_ref.extractall('pretrained/glove')

from gensim.utils import simple_preprocess

def get_split_glove_embedding(glove_source = 'pretrained/glove/glove.twitter.27B.200d.txt'):
    '''Get split data, 
    get glove model and make embeddings from data.
    '''
    #Load glove model
    print("Loading GloVe model")
    glove_model = {}
    with open(glove_source,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print("Done loading GloVe model\n")

    unkown_character = np.zeros( len( glove_model[ list( glove_model.keys() )[0] ] ) )

    #Load Data
    X_train, y_train, X_validate, y_validate, X_test, y_test = load_split_data()

    print("Embedding data")

    def get_sentence_embedding(sentence):
        sentence_embedding = []
        for word in simple_preprocess(sentence):
            if word in glove_model:
                sentence_embedding.append(glove_model[word])
            else:
                sentence_embedding.append(unkown_character)
        return sentence_embedding

    X_train = [np.array(get_sentence_embedding(sentence)) for sentence in X_train]
    X_validate = [np.array(get_sentence_embedding(sentence)) for sentence in X_validate]
    X_test = [np.array(get_sentence_embedding(sentence)) for sentence in X_test]

    print("Done Embedding data")
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


#------------------- Scoring -------------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def get_label(y):
    '''Get the associated label for the two classes outputed. Go from out size of 2 to 1'''
    return y[:, 0] < y[:, 1]

def score(y, target_y):
    y = get_label(y)
    target_y = get_label(target_y)
    return f"accuracy : {accuracy_score(target_y, y)} | precision : {precision_score(target_y, y)} | recall : {recall_score(target_y, y)}"