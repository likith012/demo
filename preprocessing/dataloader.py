#%%
import torch
import time
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"][:,0,:].unsqueeze(1)
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
        else:
            self.x_data = X_train
        if isinstance(y_train,np.ndarray):
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.y_data = y_train
        self.len = X_train.shape[0]
        self.config = config


    def __getitem__(self, index):
        return self.x_data[index].float(),self.y_data[index]

    def __len__(self):
        return self.len

def ft_data_generator(x_dat,y_dat,configs):
    
    subjects = len(x_dat) # subjects*epochs*channels*3000
    
    train_subjects = int(configs.train_split_ratio * subjects)
    
    assert train_subjects!=subjects
    
    train_ds = {}
    
    train_ds['samples'] = x_dat[0]
    train_ds['labels'] = y_dat[0]
    
    for i in range(train_subjects-1):
        train_ds['samples'] = torch.cat((train_ds['samples'],x_dat[1+i]),axis=0)
        train_ds['labels'] = torch.cat((train_ds['labels'],y_dat[1+i]),axis=0)
    
    train_ds = Load_Dataset(train_ds, configs)
    train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=configs.workers,pin_memory=True,persistent_workers=True)    

    
    test_ds = {}
    
    test_ds['samples'] = x_dat[train_subjects]
    test_ds['labels'] = y_dat[train_subjects]
    
    for i in range(train_subjects+1,subjects):
        test_ds['samples'] = torch.cat((test_ds['samples'],x_dat[i]))
        test_ds['labels'] = torch.cat((test_ds['labels'],y_dat[i]))
    

    test_ds = Load_Dataset(test_ds, configs)

    test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=configs.workers,pin_memory=True,persistent_workers=True)


    return train_loader,test_loader