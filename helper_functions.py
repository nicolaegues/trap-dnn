
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_x = self.X[idx]
        sample_y = self.y[idx]

        # Apply transformation if provided
        if self.transform:
            sample_x = self.transform(sample_x)
        
        return sample_x, sample_y


def plot_4_ims(og_diffr, pred_diffr, og_amp, pred_amp, dir):
    fig, axes = plt.subplots(1, 4, figsize = (10, 3))
    axes[0].imshow(og_diffr)
    axes[0].set_title("OG Diffraction pattern")
    axes[1].imshow(pred_diffr)
    axes[1].set_title("Pred Diffraction pattern")

    axes[2].imshow(og_amp, cmap = "twilight")
    axes[2].set_title("OG Phase")
    axes[3].imshow(pred_amp, cmap = "twilight")
    axes[3].set_title("Pred Phase")

    for ax in axes: 
        ax.set_axis_off()

    plt.savefig(dir)
    plt.close()

