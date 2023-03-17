'''This modules contains the baseplate methods to load train, test and validation data.
'''

import pandas as pd
import numpy as np


def _prepare_data(dataframe: pd.DataFrame):
    '''Split the dataframe in inputs and labels.
    The labels are divided in two columns, corresponding to each class
    '''
    dataframe['appro'] = 1 - dataframe["class"]
    dataframe['inappro'] = dataframe["class"]

    return (np.array(dataframe["text"]), np.array(dataframe[["appro", "inappro"]]))


def load_split_data():
    '''Load the full set of data splitted into train, test and validation
    '''
    X_train, y_train = _prepare_data(pd.read_csv("data/train.csv"))
    X_validate, y_validate = _prepare_data(pd.read_csv("data/validate.csv"))
    X_test, y_test = _prepare_data(pd.read_csv("data/test.csv"))
    return (X_train, y_train, X_validate, y_validate, X_test, y_test)


def get_text_data():
    return pd.read_csv("data/full.csv")["text"]