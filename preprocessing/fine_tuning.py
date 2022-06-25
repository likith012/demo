import glob
import os
import torch
import numpy as np

from edf_to_np import get_data_from_edf
from config import Config
from dataloader import ft_data_generator
from tqdm import tqdm
from trainer import sleep_ft
from prediction import predict


# get the training data
train_data_dir = "/scratch/physionet-sleep-data/"
psg_fnames = glob.glob(os.path.join(train_data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(train_data_dir, "*Hypnogram.edf"))
psg_fnames.sort()
ann_fnames.sort()
print("Number of PSG files: ", len(psg_fnames))
print("Number of annotation files: ", len(ann_fnames))
psg_fnames = np.asarray(psg_fnames)
ann_fnames = np.asarray(ann_fnames)

tot_x, tot_y = [], []

for i in range(2):

    x_dat, y_dat, sf = get_data_from_edf(psg_fnames[i], ann_fnames[i])

    tot_x.append(torch.tensor(x_dat))
    tot_y.append(torch.tensor(y_dat))

# create config object
config = Config()

# get the dataloaders
train_dl, test_dl = ft_data_generator(tot_x, tot_y, config)

# create the training model
model = sleep_ft(config, train_dl, test_dl)
model.load()
# start the training
finetuned_model = model.fit()

predict(finetuned_model, psg_fnames[12], config.device)
