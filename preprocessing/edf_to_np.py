import argparse
import glob
import math
import ntpath
import os
import shutil


from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import butter,lfilter
from mne.io import concatenate_raws, read_raw_edf

import dhedfreader

data_dir = "/scratch/physionet-sleep-data/"

# Read raw and annotation EDF files
psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
psg_fnames.sort()
ann_fnames.sort()
print("Number of PSG files: ", len(psg_fnames))
print("Number of annotation files: ", len(ann_fnames))
psg_fnames = np.asarray(psg_fnames)
ann_fnames = np.asarray(ann_fnames)

def get_data_from_edf(edf_path,ann_path):
    # Label values
    W,N1,N2,N3,REM,UNKNOWN = 0,1,2,3,4,5

    stage_dict = {"W": W,"N1": N1,"N2": N2,"N3": N3,"REM": REM,"UNKNOWN": UNKNOWN}

    class_dict = {0: "W",1: "N1",2: "N2",3: "N3",4: "REM",5: "UNKNOWN"}

    ann2label = {"Sleep stage W": 0,"Sleep stage 1": 1,"Sleep stage 2": 2,"Sleep stage 3": 3,"Sleep stage 4": 3,"Sleep stage R": 4,"Sleep stage ?": 5,"Movement time": 5}

    EPOCH_SEC_SIZE = 30
    
    # Select channel
    all_picks = ['EEG Fpz-Cz',
            'EEG Pz-Oz',
            'EOG horizontal',
            'Resp oro-nasal',
            'EMG submental',
            'Temp rectal',
            'Event marker']
    select_ch = all_picks[: 2]    
    
    raw = read_raw_edf(edf_path, preload=True, stim_channel=None)
    sampling_rate = raw.info['sfreq']
    raw_ch_df = raw.to_data_frame()[select_ch]
       
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))
    
    # Get raw header
    f = open(edf_path, 'r', errors='ignore')
    reader_raw = dhedfreader.BaseEDFReader(f)
    reader_raw.read_header()
    h_raw = reader_raw.header
    f.close()
    raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")
    
    # Read annotation and its header
    f = open(ann_path, 'r', errors='ignore')
    reader_ann = dhedfreader.BaseEDFReader(f)
    reader_ann.read_header()
    h_ann = reader_ann.header
    _, _, ann = zip(*reader_ann.records())
    f.close()
    ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")
    
    # Assert that raw and annotation files start at the same time
    assert raw_start_dt == ann_start_dt
    
    # Generate label and remove indices
    remove_idx = []    # indicies of the data that will be removed
    labels = []        # indicies of the data that have labels
    label_idx = []

    for a in ann[0]:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char)
        label = ann2label[ann_str[2:-1]]
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise Exception("Something wrong")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=np.int32) * label
            labels.append(label_epoch)
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int32)
            label_idx.append(idx)

        else:
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=np.int32)
            remove_idx.append(idx)

    labels = np.hstack(labels)

    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch_df))

    # Select only the data with labels
    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)

    # Remove extra index
    if len(label_idx) > len(select_idx):
        extra_idx = np.setdiff1d(label_idx, select_idx)
        # Trim the tail
        if np.all(extra_idx > select_idx[-1]):

            n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
            if n_label_trims!=0:

                labels = labels[:-n_label_trims]

    # Remove movement and unknown stages if any
    raw_ch = raw_ch_df.values[select_idx]

    # Verify that we can split into 30-s epochs
    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise Exception("Something wrong")
    n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

    # Get epochs and their corresponding labels
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    assert len(x) == len(y)
    
    x = np.transpose(x,(0,2,1))
    x = np.expand_dims(x[:,0,:],axis=1)
    
    return x,y,sampling_rate