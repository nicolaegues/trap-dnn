
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, X, c, y, transform=None):
        self.X = X
        self.c = c
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_x = self.X[idx]
        sample_c = self.c[idx]
        sample_y = self.y[idx]

        # Apply transformation if provided
        if self.transform:
            sample_x = self.transform(sample_x)
        
        return sample_x, sample_c, sample_y

def trap_amplitude_loss(batch_size, pred_magn, coords):
    """
    Computes the MSE between predicted amplitude and ideal target (1.0) at trap coordinates.

    Args:
        pred_magn (Tensor): Predicted field magnitudes of shape (Batchsize, 1, H, W)
        coords (Tensor): shape (Batchsize, no_traps, 2) with x,y coordinates of the traps

    Returns:
        Scalar loss 
    """

    loss = 0.0
    no_traps = coords.shape[1]
    
    for i in range(batch_size):
        for t in range(no_traps):

            a = pred_magn[i][..., coords[i][t][1], coords[i][t][0]]
            loss += F.mse_loss(a, torch.tensor([1.0]))

    return loss

def plot_4_ims(og_magn, pred_magn, og_phase, pred_phase, coords, dir):

    fig, axes = plt.subplots(1, 3, figsize = (12, 3))
    im1 = axes[0].imshow(og_magn)
    axes[0].set_title("Original Acoustic Field Magnitude")
    im2 = axes[1].imshow(pred_magn)
    axes[1].set_title("Predicted Acoustic Field Magnitude")
    axes[1].plot(coords[0][0], coords[0][1], "ro", markersize = 2)
    axes[1].plot(coords[1][0], coords[1][1], "ro", markersize = 2)

    # im3 = axes[2].imshow(og_phase, cmap = "twilight")
    # axes[2].set_title("OG Phase")
    im3 = axes[2].imshow(pred_phase, cmap = "twilight")
    axes[2].set_title("Predicted Phases")

    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(im1, ax = axes[0], shrink = 0.9)
    fig.colorbar(im2, ax = axes[1], shrink = 0.9)
    fig.colorbar(im3, ax = axes[2], shrink = 0.9)
    #fig.colorbar(im4, ax = axes[3], shrink = 0.7)

    fig.savefig(dir, dpi = 300)
    plt.close()

