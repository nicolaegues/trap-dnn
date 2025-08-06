"""
Acoustic Autoencoder Testing Script
Author: Nicola Egües
Date: 30.05.2025
"""

#================================== Imports ==================================
import torch
from acoustic_autoencoder import recon_model
import numpy as np
from utils import CustomDataset, plot_4_ims
from torch.utils.data import DataLoader
import os

#import matplotlib.pyplot as plt

#================================== Experiment Configuration ==================================

#root_dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/"
root_dir = os.getcwd()

# Test data directory
data_dir = root_dir + "/data/binary_traps_moving_closer/"
#data_dir = root_dir + "/data/perfect_binary_traps/test/"
#data_dir = root_dir + "data/random/test/"
#data_dir = root_dir + "data/twin_overfit/train/"  #then also do acoustic_vortex (single). and random test. 

# Experiment folder where the model's state dictionary is located 
#exp = "KEEP_exp_Thu-03-Jul-2025-at-03-53-57PM"
exp = "z_IASA_exp_Tue-05-Aug-2025-at-01-50-16PM"
#exp  = "z_OG_exp_Tue-05-Aug-2025-at-01-55-40PM"
#exp = "z_IASA_noamp_exp_Tue-05-Aug-2025-at-03-57-38PM"
#exp = "z_OG_noamp_exp_Tue-05-Aug-2025-at-03-58-57PM"

exp_dir = f"{root_dir}/experiments/{exp}/"

test_figs_dir = f"{exp_dir}test_figs/"
os.makedirs(test_figs_dir, exist_ok=True)

max_figs_to_test = 20

print("-"*100)
print(f"Experiment directory: {exp_dir}")
print("-"*100)

#================================== Load and Normalise Data ==================================

#Load target field magnitudes and normalise per sample (in case they are not normalised already)
target_pattern = np.load(data_dir + "acoustic_traps.npy")[:max_figs_to_test]
max_vals = np.amax(np.abs(target_pattern), axis=(1, 2), keepdims=True)
target_pattern = target_pattern/max_vals
target_pattern = torch.Tensor(target_pattern[:, np.newaxis ])

# Load trap coordinates (for loss computation later) and source phases (only for visualisation later)
trap_coords = np.load(data_dir + "trap_coords.npy")[:max_figs_to_test]
#trap_coords = np.zeros(shape = (target_pattern.shape[0], 2, 2))
source_phases = np.zeros(shape = (target_pattern.shape[0], target_pattern.shape[2], target_pattern.shape[3]))
#source_phases =  np.load(data_dir + "acoustic_phases.npy")[:no_figs_to_test]
source_phases = torch.Tensor(source_phases[:, np.newaxis ])

# Create DataLoader
batch_size = 1
dataset = CustomDataset(target_pattern, trap_coords, source_phases)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

#================================== Model Initialiation ==================================
# Initialise model and load its state dictionary in from the experiment's folder
model = recon_model()
model.load_state_dict(torch.load(f"{exp_dir}final_model.pth"))
model.eval()

for i, (magn_batch, trap_coords_batch, phase_batch) in enumerate(test_loader): 
     
     with torch.no_grad(): 
        pred_magn, pred_amp = model(magn_batch)
            
        plot_4_ims(magn_batch.detach()[0][0], pred_magn.detach()[0][0], 
                   phase_batch.detach()[0][0], pred_amp.detach()[0][0], 
                   trap_coords_batch.detach()[0], dir = f"{test_figs_dir}{i}")
        arr = np.array([np.array(magn_batch.detach()[0][0]), np.array(pred_magn.detach()[0][0]), 
                        np.array(phase_batch.detach()[0][0]), np.array(pred_amp.detach()[0][0])])
        np.save(f"{test_figs_dir}{i}", arr)

        coords_arr = trap_coords_batch.detach()[0]
        np.save(f"{test_figs_dir}coords_{i}", coords_arr)

