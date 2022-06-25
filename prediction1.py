from mne.io import read_raw_edf
from tqdm import tqdm
import numpy as np
import torch
import mne
from model import predict_model
from config import Config
import matplotlib.pyplot as plt

def predict(model, edf_path, device, ann_file=None, channel="EEG Fpz-Cz"):
    event_mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
        "Sleep stage ?": 0,
        "Movement time": 0,
    }
    EPOCH_LEN = 30  # secs
    raw = read_raw_edf(edf_path, preload=True, stim_channel=None)
    sampling_rate = raw.info["sfreq"]
    if ann_file == None:
        raw_ch_df = raw.to_data_frame()[[channel]]
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        remove_idx = int(raw_ch_df.shape[0] % (EPOCH_LEN * sampling_rate))
        raw_ch_df = raw_ch_df[: raw_ch_df.shape[0] - remove_idx]
        assert raw_ch_df.shape[0] % (EPOCH_LEN * sampling_rate) == 0
    else:
        ann = mne.read_annotations(ann_file)
        raw.set_annotations(ann, emit_warning=False)
        ann, _ = mne.events_from_annotations(
            raw, event_id=event_mapping, chunk_duration=EPOCH_LEN
        )
        raw_ch_df = raw.to_data_frame()[[channel]]
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))
        ann = ann[:, 2]
    x_dat = torch.tensor(np.array(raw_ch_df).reshape(-1, 1, 3000))
    x_dat = x_dat.to(device).float()
    y_preds = []
    model.eval()
    for epoch in tqdm(x_dat):
        ret = model(epoch.unsqueeze(0)).detach().cpu().numpy()
        ret = ret.argmax(axis=1)
        y_preds.append(ret[0])
    # print(sum(y_preds==ann)/len(ann))
    return np.array(y_preds)
# create the training model
config = Config()
chk_path = config.save_path
edf_path = r"D:\Desktop/EDF_data/ST7022J0-PSG.edf"
model = predict_model(config).to(config.device)
outs = predict(model, edf_path, config.device)
outs.tofile('predictions.csv', sep = ',')
