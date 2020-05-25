import os
import sys
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Used to import libraries from an absolute path starting with the project's root
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

# Local imports
from src.dataset.similarityDataset import SimilarityDataset

def padding_collate(batch):
    """
        Used as a PyTorch collate_fn function in PyTorch dataloaders. 
        Given a batch of vectors of shape (word_count, word_size), 
        make the word_count of each sequence uniform by doing right-side 0 padding.
        
        /!\ The sequences size between batches may vary /!\
    """
    
    max_shape_val = max(
        [
                #b[0] to get X, b[0][0] to get the first sentence of every X, b[0][1] to get the second sentence of every X
            max(b[0][0].shape[0], b[0][1].shape[0]) for b in batch
        ]
    )
    
    X1 = []
    X2 = []
    y = [] 
        
    for i in range(len(batch)):
        
        #batch[i][0] is X, batch[i][1] is y
        x1 = batch[i][0][0]
                    
        if x1.shape[0] < max_shape_val:            
            to_be_padded_shape = (max_shape_val - x1.shape[0], x1.shape[1])
            padding = torch.zeros(to_be_padded_shape)
            x1 = torch.cat((x1, padding), dim=0)
                
        x2 = batch[i][0][1]
        
        if x2.shape[0] < max_shape_val:            
            to_be_padded_shape = (max_shape_val - x2.shape[0], x2.shape[1])
            padding = torch.zeros(to_be_padded_shape)
            x2 = torch.cat((x2, padding), dim=0)
    
        X1.append(x1)
        X2.append(x2)
                
        y.append([batch[i][1]])
    
    X1 = torch.stack(X1)
    X2 = torch.stack(X2)
    
    return (X1, X2), torch.FloatTensor(y)

def model_and_titles_to_distance_dataset(model, dataloader):
    """
    Given a similarity learning model and a dataloader, transforms the data of the dataloader in the shape of a distance dataset.
    The distances are computed using the model. We then use the target variables to train a linear model to classify the distances
    to 2 different classes: similar or dissimilar.
    """



    X = []
    y = []
    
    n = 1
    
    total_duration = 0
    total_steps = len(dataloader)
    
    model.eval()
    
    for local_batch, local_labels in dataloader:
        
        t0 = time.time()
        
        # Transfer to GPU
        local_batch_X1, local_batch_X2, local_labels = local_batch[0].to(device), local_batch[1].to(device), local_labels.to(device)

        preds1, preds2 = model(local_batch_X1, local_batch_X2)
        
        with torch.no_grad():
            distances = torch.dist(preds1, preds2, 2)
            # Transfering distances and labels to cpu
            distances_cpu = distances.cpu().numpy().reshape(-1, 1)
            labels = torch.flatten(local_labels).cpu().numpy().reshape(-1, 1)
            # Fitting logreg
            X.append(distances_cpu)
            y.append(labels)
        
        duration = time.time() - t0
        
        total_duration += duration
        
        per_step_mean_duration = total_duration / n
        rest_of_time = per_step_mean_duration * (total_steps)
        
        n+=1
        print(f"\r{n}-{total_steps} (ETA: {total_duration}/{rest_of_time}s)", end="")
        
    model.train()
        
    X = np.array(X).flatten().reshape(-1, 1)
    y = np.array(y).flatten()
    
    return X, y


BATCH_SIZE = 8
EMBEDDING_DIM = 20
EPOCHS = 50
TRAIN = 0.8
TEST = 0.1
VAL = 0.1
SHUFFLE = True
SEED = 42
LR = 1e-3

if __name__ == "__main__":

    dataset = SimilarityDataset()

    # add dataloaders + splits + training functions
    # add model and loss as well

    print("#########################################")
    print("#                PART I.                #")
    print("#                                       #")
    print("# -> Initializing data objects          #")
    print("#    for training, testing and          #")
    print("#    validation.                        #")
    print("#                                       #")
    print("#########################################")
    
    # Data preparation 
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    val_split = int(np.floor(VAL * dataset_size))
    test_split = int(np.floor(TEST * dataset_size))

    if SHUFFLE:
        np.random.seed(SEED)
        np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[val_split+test_split:], indices[:val_split], indices[:val_split+test_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    num_train_batch = int(np.ceil(TRAIN * dataset_size / BATCH_SIZE))
    num_val_batch = int(np.ceil(VAL * dataset_size / BATCH_SIZE))
    num_test_batch = int(np.ceil(TEST * dataset_size / BATCH_SIZE))
    
    print("Creating dataloaders..")
    
    dataloader_train = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        collate_fn = collate_fn,
        sampler = train_sampler
    )

    dataloader_val = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        collate_fn = collate_fn,
        sampler = val_sampler
    )

    dataloader_test = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        collate_fn = collate_fn,
        sampler = test_sampler
    )
    
    
    print("#########################################")
    print("#                PART II.               #")
    print("#                                       #")
    print("# -> Training siamese networks with     #")
    print("#    contrastive loss to generate       #")
    print("#    latent representation of sequences #")
    print("#    and calculate distances between    #")
    print("#    them.                              #")
    print("#                                       #")
    print("#########################################")

    model = TextSimilarityLSTM(embedding_dim = EMBEDDING_DIM)
    model.cuda()
    model.train()
    
    contrastive_loss = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

    
    for i in range(EPOCHS):

        print(f"Epochs {i}")
        n = 1
        total_loss = 0
        total_duration = 0
        total_timesteps = len(dataloader_train)
        
        for local_batch, local_labels in dataloader_train:
            model.zero_grad()

            t0 = time.time()
            
            # Transfer to GPU
            local_batch_X1, local_batch_X2, local_labels = local_batch[0].to(device), local_batch[1].to(device), local_labels.to(device)

            preds1, preds2 = model(local_batch_X1, local_batch_X2)

            # Compute the loss, gradients, and update the parameters by
            #loss = loss_function(preds, local_labels)
            loss = contrastive_loss(preds1, preds2, local_labels)
            loss.backward()
            optimizer.step()

            # Statistics to follow progress
            total_loss += loss.item()
            duration = time.time() - t0
            total_duration += duration
            total_duration = round(total_duration, 2)
            
            estimated_duration_left = round((total_duration / n) * (total_timesteps), 2)

            print(f"\r Epochs {i} - Loss: {total_loss/n} - Acc: {0} - Batch: {n}/{num_train_batch} - Dur: {total_duration}s/{estimated_duration_left}s", end="")
            n+=1

        print("\n")
        
        # End of epochs validation
        
        n = 1
        total_loss = 0
        with torch.no_grad():
            model.eval()
            for local_batch, local_labels in dataloader_val:
                # Transfer to GPU
                local_batch_X1, local_batch_X2, local_labels = local_batch[0].to(device), local_batch[1].to(device), local_labels.to(device)

                preds1, preds2 = model(local_batch_X1, local_batch_X2)

                loss = contrastive_loss(preds1, preds2, local_labels)

                total_loss += loss.item()

                acc=0
                print(f"\r Epochs {i} - Val_loss: {total_loss/n} - Batch: {n}/{num_val_batch}", end="")
                n+=1
            
            if to_save:
                date = datetime.now().strftime("%m_%d_%H_%M_%S")
                torch.save(model.state_dict(), f"siamese_lstm_sequence_{date}_epoch{i}.pt")
            
            model.train()
        print("\n---")
    
    if to_save:
        date = datetime.now().strftime("%m_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"siamese_lstm_sequence_{date}_final.pt")
    
    print("########################################")
    print("#               PART III.              #")
    print("#                                      #")
    print("# -> Using the trained network to      #")
    print("#    calculate distances between       #")
    print("#    titles using their latent         #")
    print("#    representation extracted from     #")
    print("#    the siamese network.              #")
    print("#                                      #")
    print("#    These distances will then be      #")
    print("#    used to train a linear            #")
    print("#    classifier to say if a given      #")
    print("#    distance corresponds to two       #")
    print("#    similar titles                    #")
    print("#                                      #")
    print("########################################")

    dataloader_train = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        collate_fn = collate_fn,
        sampler = train_sampler
    )

    dataloader_val = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        collate_fn = collate_fn,
        sampler = val_sampler
    )

    dataloader_test = torch.utils.data.dataloader.DataLoader(
        dataset = dataset,
        collate_fn = collate_fn,
        sampler = test_sampler
    )

    _debug("Generating training dataset...")
    X_train, y_train = model_and_titles_to_distance_dataset(model, dataloader_train)
    
    _debug("Generating validation dataset...")
    X_test, y_test = model_and_titles_to_distance_dataset(model, dataloader_test)
    
    _debug("Generating test dataset...")
    X_val, y_val = model_and_titles_to_distance_dataset(model, dataloader_val)

    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)

    train_acc = logreg.score(X_train, y_train)
    val_acc = logreg.score(X_val, y_val)
    test_acc = logreg.score(X_test, y_test)

    print(f"\nFinal perfs: {train_acc} - {val_acc} - {test_acc}")