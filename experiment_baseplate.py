import pandas as pd
import numpy as np


def prepare_data(dataframe: pd.DataFrame):
    '''Split the dataframe in inputs and labels.
    The labels are divided in two columns, corresponding to each class
    '''
    dataframe['appro'] = 1 - dataframe["class"]
    dataframe['inappro'] = dataframe["class"]

    return (np.array(dataframe["text"]), np.array(dataframe[["appro", "inappro"]]))


X_train, y_train = prepare_data(pd.read_csv("data/train.csv"))
X_validate, y_validate = prepare_data(pd.read_csv("data/validate.csv"))
X_test, y_test = prepare_data(pd.read_csv("data/test.csv"))
