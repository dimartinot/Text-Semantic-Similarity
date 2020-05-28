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


class SimilarityVectorizedDataset(Dataset):
    """
        PyTorch compatible dataset of product titles.
        Wrapper of the productDataset.
    """

    def __init__(self, dataset_path = None):

        self._dataset = pd.read_pickle(DEFAULT_DATASET_DS_PATH) if dataset_path is None else pd.read_pickle(dataset_path)
        """ Pandas dataframe variable acting as the dataset """            

        self._y = self._dataset["is_duplicate"].apply(lambda cell: 1 if str(cell) == "1" else 0).to_numpy()  
        """ Variable holding the class of the tuple. If products are similar, then the class is 1. It is 0 otherwise """

        self._x1 = self._dataset["question1"].to_numpy()
        """ Variable 1 holding the tuple sentences acting as input of the model """
        self._x2 = self._dataset["question2"].to_numpy()
        """ Variable 2 holding the tuple sentences acting as input of the model """

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
        sequence1, sequence2 = self._x1[idx], self._x2[idx]

        sequence1 = torch.FloatTensor(sequence1).squeeze(1)
        sequence2 = torch.FloatTensor(sequence2).squeeze(1)

        # if two sequences are different sizes, perform zero padding of the smallest
        if (sequence1.shape != sequence2.shape):
            if (sequence1.shape[0] < sequence2.shape[0]):

                to_be_padded_shape = (sequence2.shape[0] - sequence1.shape[0], sequence2.shape[1])
                padding = torch.zeros(to_be_padded_shape)
                sequence1 = torch.cat((sequence1, padding), dim=0)

            else:

                to_be_padded_shape = (sequence1.shape[0] - sequence2.shape[0], sequence2.shape[1])
                padding = torch.zeros(to_be_padded_shape)
                sequence2 = torch.cat((sequence2, padding), dim=0)

        x = (sequence1, sequence2)

        # Part 2 : Retrieving the class of the sequence
        Y = self._y[idx]

        return (x, Y)