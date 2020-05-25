import os
import sys
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Used to import libraries from an absolute path starting with the project's root
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.dataset.similarityDataset import SimilarityDataset

if __name__ == "__main__":

    dataset = SimilarityDataset()

    # add dataloaders + splits + training functions
    # add model and loss as well
