######################################
# SimilarityDataset.py               #
#                                    #
# V 1.0                              #
# PyTorch Dataset variable embedding #
# information about the questions    #
# pairs.                             #
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

# DS Libraries imports
import torch
from torch.utils.data import Dataset
import numpy as np 
import pandas as pd

# Local imports
from src.preprocessing.word2Vec import Word2VecModel

# Constants
SEED = 42
CURRENT_FILE_FOLDER = os.path.split(os.path.realpath(__file__))[0]
DEFAULT_DATASET_DS_PATH = os.path.join(CURRENT_FILE_FOLDER, os.path.join("../..","data/vectorized_dataset.pkl"))

np.random.seed(SEED)

def get_random_unmatching_index(random_max_val, indexes_to_avoid):
    """
        Function used to generate negative matching questions
    """
    val = np.random.randint(0, random_max_val)
    while val in indexes_to_avoid:
        val = np.random.randint(0, random_max_val)
    return val

def padding(data, new_shape):
    """
        Given a data of shape (s1, s2) and new_shape a tuple (s3, s2).
    """
    to_be_padded_shape = (new_shape[0] - data.shape[0], data.shape[1])
    zeros = torch.zeros(to_be_padded_shape)
    return torch.cat((data, zeros), dim=0)


class MultiLossSimilarityVectorizedDataset(Dataset):
    """
        PyTorch compatible dataset of product titles.
        Wrapper of the productDataset.
    """

    def __init__(self, dataset_path = None):

        self._dataset = pd.read_pickle(DEFAULT_DATASET_DS_PATH) if dataset_path is None else pd.read_pickle(dataset_path)
        """ Pandas dataframe variable acting as the dataset """            
        self._dataset = self._dataset[self._dataset["is_duplicate"] == 1]

        self._y = self._dataset["is_duplicate"].to_numpy()  
        """ Variable holding the class of the tuple. If products are similar, then the class is 1. It is 0 otherwise """

        self._x1 = self._dataset["question1"].to_numpy()
        """ Variable 1 holding the tuple sentences acting as input of the model """
        self._positive = self._dataset["question2"].to_numpy()
        """ Variable 2 holding the tuple sentences acting as input of the model """
        self._negative1 = [
            get_random_unmatching_index(len(self._x1), [idx]) for idx in range(len(self._x1))
        ]
        self._negative2 = [
            get_random_unmatching_index(len(self._x1), [idx, self._negative1[idx]]) for idx in range(len(self._x1))
        ]

    def __len__(self):
        """
        Returns the size of the dataset when called through the *len(..)* function.
        """
        return len(self._dataset)

    def __getitem__(self, idx):
        """
        When the dataset is accessed using dataset[idx], returns a tuple (x, Y) where:
         - x is itself a tuple (sequence1, sequence2)
         - y is the class saying whether the two sequences are similar or not
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Part 1: Retrieving the tuple of sequences
        sequence1, sequence2, sequence3, sequence4 = self._x1[idx], self._positive[idx], self._x1[self._negative1[idx]], self._x1[self._negative2[idx]]

        sequence1 = torch.FloatTensor(sequence1).squeeze(1)
        sequence2 = torch.FloatTensor(sequence2).squeeze(1)
        sequence3 = torch.FloatTensor(sequence3).squeeze(1)
        sequence4 = torch.FloatTensor(sequence4).squeeze(1)

        max_shape = (
            max([sequence1.shape[0], sequence2.shape[0], sequence3.shape[0], sequence4.shape[0]]),
            sequence1.shape[1]
        )

        sequence1 = padding(sequence1, max_shape)
        sequence2 = padding(sequence2, max_shape)
        sequence3 = padding(sequence3, max_shape)
        sequence4 = padding(sequence4, max_shape)
        
        x = (sequence1, sequence2, sequence3, sequence4)

        # Part 2 : Retrieving the class of the sequence
        Y = self._y[idx]

        return (x, Y)