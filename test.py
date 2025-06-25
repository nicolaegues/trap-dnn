

import torch
from acoustic_DNN.acoustic_autoencoder import recon_model
import numpy as np
from helper_functions import CustomDataset, plot_4_ims
from torch.utils.data import DataLoader
import os

exp = "KEEP_exp_Wed-25-Jun-2025-at-02-54-45PM"

dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/acoustic_DNN/"
#data directory
data_dir = dir + "data/twin_trap/train/"
exp_dir = f"{dir}experiments/{exp}/"

test_figs_dir = f"{exp_dir}test_figs_3/"
os.makedirs(test_figs_dir, exist_ok=True)

test_pattern = np.load(data_dir + "acoustic_traps.npy")
test_pattern = torch.Tensor(test_pattern[:, np.newaxis ])
source_phases =  np.load(data_dir + "acoustic_phases.npy") #just for visualisation
source_phases = torch.Tensor(source_phases[:, np.newaxis ])


dataset = CustomDataset(test_pattern, source_phases)
test_loader = DataLoader(dataset, batch_size=10, shuffle = True)


model = recon_model()
model.load_state_dict(torch.load(f"{exp_dir}final_model.pth"))
model.eval()


for b, (diffr_batch, aperture_batch) in enumerate(test_loader): 
     with torch.no_grad(): 

        pred_diffr, pred_amp = model(diffr_batch)

        if b == len(test_loader)-1: 
            for i in range(len(diffr_batch)): 
                plot_4_ims(diffr_batch[i][0], pred_diffr[i][0], aperture_batch[i][0], pred_amp[i][0], dir = f"{test_figs_dir}{i}")
