
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import numpy as np
import datetime
import os
import time
import json

from autoencoder import recon_model
from helper_functions import CustomDataset, plot_4_ims

"""
# (early stopping thresh)
# learning rate scheduler?, loss scaling for AMP?, 
current params (epochs = 3, batchsize = 8, lr = 0.01 overfit a single square aperture)
"""

writer = None
#writer = SummaryWriter() #terminal: python -m tensorboard.main --logdir=runs
plot_progression = True

dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/"

#data directory
data_dir = dir + "data/one_square/"

#experiment directory
current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
exp_dir = f"{dir}experiments/exp_{current_datetime}/"
final_figs_dir = f"{exp_dir}final_figs/"
os.makedirs(final_figs_dir, exist_ok=True)
progression_figs_dir = f"{exp_dir}progression_figs/"
os.makedirs(progression_figs_dir, exist_ok=True)

train_intensities = np.load(data_dir + "intensities.npy")[:1000]
train_intensities = torch.Tensor(train_intensities[:, np.newaxis ])
apertures =  np.load(data_dir + "apertures.npy")[:1000] #just for visualisation
apertures = torch.Tensor(apertures[:, np.newaxis ])


dataset = CustomDataset(train_intensities, apertures)
train_set, val_set = train_test_split(dataset, test_size=0.2)

batch_size = 8
train_loader  = DataLoader(train_set, batch_size=batch_size, shuffle = True)
val_loader  = DataLoader(val_set, batch_size=batch_size, shuffle = True)

model = recon_model()
epochs = 3
#lr = 0.005
lr = 0.01

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()


experiment_summary = {
    "data_dir": data_dir,
    "model": "version with nn.upsample",
    "learning_rate": lr,
    "batch_size": batch_size,
    "epochs": epochs,
    "train_size": len(train_set),
    "val_size": len(val_set),
    "loss_function": "L1Loss",
    "total_time": 0,
    "time_per_train_epoch": 0
}

metrics = dict([(f"Epoch {epoch+1}", {"training_loss": [], "validation_loss": []}) for epoch in range(epochs)])

train_batches_per_epoch = len(train_set)//batch_size
no_desired_figs_per_epoch = 100
train_plot_every_n_batches = train_batches_per_epoch // no_desired_figs_per_epoch

val_batches_per_epoch = len(val_set)//batch_size
val_plot_every_n_batches = val_batches_per_epoch // 2


start_time = time.time()

for epoch in range(1, epochs + 1):
    print(f"\n Epoch {epoch}:")
    model.train()

    train_loss = []
    train_count = 0

    #for diffr_batch, amp_batch in train_loader: 
    for b, (diffr_batch, aperture_batch) in enumerate(train_loader, start = 1): 

        pred_diffr, pred_amp = model(diffr_batch)
        loss = criterion(pred_diffr, diffr_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # figure out what they did: 
        # grad_scaler.scale(loss).backward()
        # grad_scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        # grad_scaler.step(optimizer)
        # grad_scaler.update()

        if plot_progression == True: 
            if epoch == 1 and (b % train_plot_every_n_batches == 0):
                plot_4_ims(diffr_batch.detach()[0][0], pred_diffr.detach()[0][0], aperture_batch.detach()[0][0], pred_amp.detach()[0][0], dir = f"{progression_figs_dir}train_epoch{epoch}_batch{b}")

        train_loss.append(loss.item())
        if writer is not None: 
            writer.add_scalar("Loss/Train", loss.item(), b+train_batches_per_epoch*(epoch-1))

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time-start_time
        if epoch == 1:
            experiment_summary["time_per_train_epoch"] = f"{epoch_duration/60:.4f} minutes"


    # VALIDATION
    model.eval()
    val_loss = []
    for b, (diffr_batch, aperture_batch) in enumerate(val_loader): 
        with torch.no_grad(): 

            pred_diffr, pred_amp = model(diffr_batch)
            loss = criterion(pred_diffr, diffr_batch)

            val_loss.append(loss.item())

            # if plot_progression == True: 
            #     if b % val_plot_every_n_batches == 0:
            #         plot_4_ims(diffr_batch[0][0], pred_diffr[0][0], aperture_batch[0][0], pred_amp[0][0], dir = f"{progression_figs_dir}epoch{epoch}_batch{b}")


            if epoch == epochs and b == len(val_loader)-1: 
                for i in range(len(diffr_batch)): 

                    plot_4_ims(diffr_batch[i][0], pred_diffr[i][0], aperture_batch[i][0], pred_amp[i][0], dir = f"{final_figs_dir}{i}")

                    #np.save(f"{exp_dir}{i}",  np.array(pred_amp[i][0]))

            if writer is not None: 
                writer.add_scalar("Loss/Validation", loss.item(),  b+train_batches_per_epoch*(epoch-1))


            #scheduler.step(val_accuracy)
            #Early stopping based on val accuracy ?

    torch.save(model.state_dict(),f"{exp_dir}final_model.pth")

    metrics[f"Epoch {epoch}"]["training_loss"].append(train_loss)
    metrics[f"Epoch {epoch}"]["validation_loss"].append(val_loss)

    epoch_train_loss = np.array(train_loss).mean()
    print(f"Epoch {epoch} Mean Train Loss: {epoch_train_loss}")
    
    epoch_val_loss = np.array(val_loss).mean()
    print(f"Epoch {epoch} Mean Val loss: {epoch_val_loss.item()}")
    model.train()

end_time = time.time()

duration = end_time - start_time
experiment_summary["total_time"] = f"{duration/60:.4f} minutes"

with open(f"{exp_dir}metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(f"{exp_dir}experiment_summary.json", "w") as f:
    json.dump(experiment_summary, f, indent=4)

if writer is not None:
    writer.close()