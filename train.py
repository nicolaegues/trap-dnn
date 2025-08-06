"""
Acoustic Autoencoder Training Script
Author: Nicola Egües
Date: 30.05.2025
"""

"""
# (early stopping thresh)
# learning rate scheduler?, loss scaling for AMP?, 

# inlcude mean metrics at top of metrics.json
# fix loss not same size amp loss

# run on supercomp. 
# add signal to background loss term
# compare both methods in terms of static and moving results (ie how similar the phasemaps are), reduced propagation, and generalisation to multiple traps
# fix other flaws.

# update autoencoder, propagate_reduced.py, (model_visualisation)
# example data folder (w couple arrs)
# example experiment folder

"""

#================================== Imports ==================================
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np
import datetime
import os
import time
import json

from acoustic_autoencoder import recon_model
from utils import CustomDataset, trap_amplitude_loss, plot_4_ims

# import matplotlib.pyplot as plt
# from torchmetrics.image import StructuralSimilarityIndexMeasure

#================================== Experiment Configuration ==================================

#root_dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/"
root_dir = os.getcwd()

# Data directory
data_dir = root_dir + "/data/perfect_binary_traps/train/"
#data_dir = root_dir + "/data/random/train/"

# TensorBoard writer
writer = SummaryWriter() # Launch with: python -m tensorboard.main --logdir=runs
#writer = None

# Flags
plot_progression = False
include_amp_loss = True

# Training hyperparameters
epochs = 3
batch_size = 8
lr = 0.001
#amp_loss_weight = 0.05 # Weight of trap amplitude loss
amp_loss_weight = 0.1

# Experiment output folder
current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
exp_dir = f"{root_dir}/experiments/exp_{current_datetime}/"
final_figs_dir = f"{exp_dir}final_figs/"
os.makedirs(final_figs_dir, exist_ok=True)
progression_figs_dir = f"{exp_dir}progression_figs/"
if plot_progression:
    os.makedirs(progression_figs_dir, exist_ok=True)

print("-"*100)
print(f"Experiment directory: {exp_dir}")
print("-"*100)

#================================== Load and Normalise Data ==================================

#Load target field magnitudes and normalise per sample (in case they are not normalised already)
target_pattern = np.load(data_dir + "acoustic_traps.npy")
max_vals = np.amax(np.abs(target_pattern), axis=(1, 2), keepdims=True)
target_pattern = target_pattern/max_vals
target_pattern = torch.Tensor(target_pattern[:, np.newaxis ])

# Load trap coordinates (for loss computation later) and source phases (only for visualisation later)
trap_coords = np.load(data_dir + "trap_coords.npy")
#trap_coords = np.zeros(shape = (target_pattern.shape[0], 2, 2))
source_phases = np.zeros(shape = (target_pattern.shape[0], target_pattern.shape[2], target_pattern.shape[3]))
#source_phases =  np.load(data_dir + "acoustic_phases.npy") 
source_phases = torch.Tensor(source_phases[:, np.newaxis ])

# Create DataLoaders
dataset = CustomDataset(target_pattern, trap_coords, source_phases)
train_set, val_set = train_test_split(dataset, test_size=0.2, shuffle = False)
train_loader  = DataLoader(train_set, batch_size=batch_size, shuffle = False )
val_loader  = DataLoader(val_set, batch_size=batch_size, shuffle = False)


#================================== Model, Optimiser, and Loss Function Initialiation ==================================
model = recon_model()

# from torchinfo import summary
# #look at shape of typical batches in data loaders
# for idx, (X_, Y_) in enumerate(train_loader):
#     print("X: ", X_.shape)
#     print("Y: ", Y_.shape)
#     if idx >= 0:
#         break

# #model architecture summary
# summary(model,
#         input_data = X_,
#         col_names=["input_size",
#                     "output_size",
#                     "num_params"])



optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
criterion = nn.L1Loss() # MAE


#================================== Metric Tracking ==================================
experiment_summary = {
    "data_dir": data_dir,
    "model": "acoustic autoencoder",
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs,
    "train_size": len(train_set),
    "val_size": len(val_set),
    "include amplitude loss": include_amp_loss,
    "amplitude loss weight": amp_loss_weight,
    "total_time": 0,
    "time_per_train_epoch": 0
}

if include_amp_loss == True: 
    metrics = dict([(f"Epoch {epoch+1}", {"training_total_loss": [], "training_recon_loss": [], "training_amp_loss": [], "validation_total_loss": [], "validation_recon_loss": [], "validation_amp_loss": [], }) for epoch in range(epochs)])
else: 
    metrics = dict([(f"Epoch {epoch+1}", {"training_total_loss": [], "validation_total_loss": [] }) for epoch in range(epochs)])


# Desired number of progression plots per epoch
no_desired_figs_per_epoch = 25
train_batches_per_epoch = len(train_set)//batch_size
train_plot_every_n_batches = train_batches_per_epoch // no_desired_figs_per_epoch

val_batches_per_epoch = len(val_set)//batch_size
val_plot_every_n_batches = val_batches_per_epoch // 2

start_time = time.time()

#================================== Main Loop ==================================
for epoch in range(1, epochs + 1):

    print(f"\n Epoch {epoch}:")

    train_total_loss = []
    train_recon_loss = []
    train_amp_loss = []
    train_count = 0

    val_total_loss = []
    val_recon_loss = []
    val_amp_loss = []

    #====================== Training Loop ======================

    for b, (magn_batch, trap_coords_batch, phase_batch) in enumerate(train_loader, start = 1): 

        # Forward pass
        pred_magn, pred_phase = model(magn_batch)
        
        # MAE Reconstruction Loss
        recon_loss = criterion(pred_magn, magn_batch)

        # optionally include trap-amplitude loss
        if include_amp_loss == True: 
            amp_loss = trap_amplitude_loss(batch_size, pred_magn, trap_coords_batch)
            loss = recon_loss + amp_loss_weight * amp_loss
        else: 
            loss = recon_loss

        # Backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        # Generate progression plots/store data every few batches
        if plot_progression == True: #and 1 < epoch < 5: 
            if (b % train_plot_every_n_batches == 0):

                # Store the results (plot and numpy array) for the first item in the current batch
                # ----------------------------------------------------------------------------------
                # plot_4_ims(magn_batch.detach()[0][0], pred_magn.detach()[0][0], phase_batch.detach()[0][0], 
                #            pred_phase.detach()[0][0], trap_coords_batch.detach()[0][0], 
                #            dir = f"{progression_figs_dir}train_epoch{epoch}_batch{b}")
                # arr = np.array([np.array(magn_batch.detach()[0][0]), np.array(pred_magn.detach()[0][0]),
                #                 np.array(phase_batch.detach()[0][0]), np.array(pred_phase.detach()[0][0])])
                # np.save(f"{progression_figs_dir}train_epoch_{epoch}_batch{b}", arr)


                # this is to track the progression (loss and plots) of the SAME input at each stage. 
                # ----------------------------------------------------------------------------------
                model.eval()
                test = val_set[6][0][0]
                test = test[np.newaxis, np.newaxis, :]
                with torch.no_grad(): 
                    pred_magn, pred_phase = model(test)
                    temp_loss = criterion(pred_magn, test)
                    plot_4_ims(test.detach()[0][0], pred_magn.detach()[0][0], phase_batch.detach()[0][0], 
                               pred_phase.detach()[0][0], trap_coords_batch.detach()[0], 
                               dir = f"{progression_figs_dir}loss_{temp_loss.detach():.4f}_train_epoch{epoch}_batch{b}-jpg")
                model.train()


        train_total_loss.append(loss.item())
        if include_amp_loss == True:
            train_recon_loss.append(recon_loss.item())
            train_amp_loss.append(amp_loss.item())

        if writer is not None: 
            writer.add_scalar("Loss/Train", loss.item(), b+train_batches_per_epoch*(epoch-1))

        # Estimate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time-start_time
        if epoch == 1:
            experiment_summary["time_per_train_epoch"] = f"{epoch_duration/60:.4f} minutes"


    #====================== Validation Loop ======================
    model.eval()
    for b, (magn_batch, trap_coords_batch, phase_batch) in enumerate(val_loader): 

        with torch.no_grad(): 
            pred_magn, pred_phase = model(magn_batch)

            recon_loss = criterion(pred_magn, magn_batch)
            if include_amp_loss == True: 
                amp_loss = trap_amplitude_loss(batch_size, pred_magn, trap_coords_batch)
                loss = recon_loss + amp_loss_weight * amp_loss
            else: 
                loss = recon_loss

            # Store results (plots and arrays) of last batch of the last epoch
            if epoch == epochs and b == len(val_loader)-1: #and b >= len(val_loader)-2:
                for i in range(len(magn_batch)): 

                    plot_4_ims(magn_batch.detach()[i][0], pred_magn.detach()[i][0], phase_batch.detach()[i][0], 
                               pred_phase.detach()[i][0], trap_coords_batch.detach()[i], dir = f"{final_figs_dir}{i}")

                    arr = np.array([np.array(magn_batch.detach()[i][0]), np.array(pred_magn.detach()[i][0]), 
                                    np.array(phase_batch.detach()[i][0]), np.array(pred_phase.detach()[i][0])])
                    
                    coords_arr = trap_coords_batch.detach()[i]
                    np.save(f"{final_figs_dir}{i}", arr)
                    np.save(f"{final_figs_dir}coords_{i}", coords_arr)

            val_total_loss.append(loss.item())
            if include_amp_loss == True:
                val_recon_loss.append(recon_loss.item())
                val_amp_loss.append(amp_loss.item())
            if writer is not None: 
                writer.add_scalar("Loss/Validation", loss.item(),  b+train_batches_per_epoch*(epoch-1))

            #scheduler.step(val_accuracy)
            #Early stopping based on loss?


    #====================== Epoch Data Collection  ======================
    torch.save(model.state_dict(),f"{exp_dir}final_model.pth")

    metrics[f"Epoch {epoch}"]["training_total_loss"].append(train_total_loss)
    metrics[f"Epoch {epoch}"]["validation_total_loss"].append(val_total_loss)
    epoch_train_total_loss = np.array(train_total_loss).mean()
    print(f"Epoch {epoch} Mean Train Total Loss: {epoch_train_total_loss}")

    if include_amp_loss == True:
        metrics[f"Epoch {epoch}"]["training_recon_loss"].append(train_recon_loss)
        metrics[f"Epoch {epoch}"]["training_amp_loss"].append(train_amp_loss)
        metrics[f"Epoch {epoch}"]["validation_recon_loss"].append(val_recon_loss)
        metrics[f"Epoch {epoch}"]["validation_amp_loss"].append(val_amp_loss)

        epoch_train_recon_loss = np.array(train_recon_loss).mean()
        epoch_train_amp_loss = np.array(train_amp_loss).mean()
        print(f"Epoch {epoch} Mean Train Reconstruction Loss: {epoch_train_recon_loss}")
        print(f"Epoch {epoch} Mean Train Trap Amplitude Loss: {epoch_train_amp_loss}")
    
    epoch_val_total_loss = np.array(val_total_loss).mean()
    print(f"\nEpoch {epoch} Mean Val Total Loss: {epoch_val_total_loss.item()}")

    if include_amp_loss == True:
        epoch_val_recon_loss = np.array(val_recon_loss).mean()
        epoch_val_amp_loss = np.array(val_amp_loss).mean()
        print(f"Epoch {epoch} Mean Val Reconstruction Loss: {epoch_val_recon_loss}")
        print(f"Epoch {epoch} Mean Val Trap Amplitude Loss: {epoch_val_amp_loss}")


#================================== Save Results ==================================
end_time = time.time()
duration = end_time - start_time
experiment_summary["total_time"] = f"{duration/60:.4f} minutes"

with open(f"{exp_dir}metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(f"{exp_dir}experiment_summary.json", "w") as f:
    json.dump(experiment_summary, f, indent=4)

if writer is not None:
    writer.close()


