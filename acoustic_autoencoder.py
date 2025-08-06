
"""
The code for the Angular Spectrum Method belongs to Barney Emmens. 

The code for the spectral layer was inspired by the code available in the GedankenNet 
repository https://github.com/PORPHURA/GedankenNet/blob/main/GedankenNet_Phase/networks/fno.py. 
The original implementation of a Fourier Neural Operator (incl. spectral layers) can be found at 
https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/spectral_convolution.py .

"""

import torch 
from torch import nn
import numpy as np
from numpy.fft import fft2, fftshift
import torch.nn.functional as F

#================================== Constants ==================================

c_w = 1480
c_p = 2340 # speed of sound in PS https://ims.evidentscientific.com/en/learn/ndt-tutorials/thickness-gauge/appendices-velocities

# Acoustic wave parameters
f = 1e6
wavelength = c_w/f
k = 2*np.pi*f/c_w

# This part below is mainly to get a realistic aperture size, and in case a discrete grid wants to be generated instead. 
# The main implementation, however, currently just simulates a square-shaped continuous source field (not discrete).

# Source Parameters 
element_width = 3e-3 # Width of one transducer element
kerf = 0.1e-3 # Gap between elements
N_elements_per_side = 7 # Square grid (7x7 elements)
pitch = element_width + kerf # Distance between centers of adjacent elements 
aperture = N_elements_per_side*pitch - 2*kerf # Full aperture width

#size of () (slighlty larger than aperture)
Lx = 1.1*aperture 

# Focal plane distance (propagation depth)
z = 1.5*aperture


def ASM(P0,dx,z):
    """
    Propagates a 2D complex pressure field from z = 0 to z = z using 
    the Angular Spectrum Method (ASM).

    The input field P0 is first transformed into the spatial frequency domain,
    multiplied by a propagation filter (transfer function), and then transformed
    back via inverse FFT. High-angle components (evanescent waves) are filtered out.

    Args:
        P0 (ndarray): Complex-valued input pressure field at the source plane.
        dx (float): Spatial resolution [m] of the input grid.
        z (float): Propagation distance [m].

    Returns:
        ndarray: Complex-valued propagated field at distance z,
                 cropped to the original shape of P0.
    """

   # Padded size for FFT
    Nk = 2**int(np.ceil(np.log2(P0.shape[2]))+1)
    kmax = 2*np.pi/dx

    # Compute spatial frequency grids
    kv = torch.fft.fftfreq(Nk)*kmax 
    kx, ky = torch.meshgrid(kv, kv, indexing='ij')
    kz = torch.sqrt((k**2 - kx**2 - ky**2).to(torch.complex64))# Allow for complex values

    # Transfer function
    H = torch.exp(-1j*kz*z)

    # Limit angular spectrum to propagating waves only
    D = (Nk-1)*dx
    kc = k*torch.sqrt(torch.tensor(0.5*(D**2)/(0.5*D**2 + z**2)))  # Angular cutoff
    H[torch.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate

    # Propagate the field
    P0_fourier = torch.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
    P_z_fourier = P0_fourier * H

    P_z = torch.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field
    P_z = P_z[..., :P0.shape[2], :P0.shape[3]]

    return P_z


nconv = 64
H = 64

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.modes = modes
        scale = 1 / in_channels
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes // 2 + 1, 2))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batchsize, x.size(1), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(x_ft, w)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x

class SpectralBlock(nn.Module):

    def __init__(self, channels, modes=16):
        super().__init__()
        self.spec_conv = SpectralConv2d(channels, channels, modes)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.prelu = nn.PReLU(channels) #try GELU?

    def forward(self, x):
        x_spec = self.spec_conv(x)
        x_conv = self.conv(x)
        return self.prelu(x + x_spec + x_conv)
    
class In(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            #nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x) 

class Down(nn.Module):
    """(convolution => [BN] => ReLU) **2 """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.PReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            #nn.PReLU(),
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module): 
   
    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.deconv_= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

        )

        self.deconv= nn.Sequential(
    
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
    
    def forward(self, x):
        return self.deconv(x)
    
class Out(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)            
        

    def forward(self, x):
        return self.out(x)

class recon_model(nn.Module):

  def __init__(self):
    super(recon_model, self).__init__()

    self.inc = In(1, nconv) 
    self.down1 = Down(nconv, nconv*2)
    self.down2= Down(nconv*2, nconv*4)
    #self.down3= Down(nconv*4, nconv*8)

    #self.spec  = FNOBlocks(nconv*8,nconv*8, n_modes=[8,8], n_layers = 1, use_channel_mlp = False)   
    self.spec_block = SpectralBlock(nconv * 4, modes=16)
    #self.spec_block2 = SpectralBlock(nconv * 4, modes=8)

    #self.up1 = Up(nconv*8, nconv*4)
    self.up2 = Up(nconv*4,  nconv*2)
    self.up3 = Up(nconv*2,  nconv)

    self.outc = Out(nconv, 1)            

    ##########################################################333

    # self.inc = nn.Conv2d(1, nconv, 3, padding=1)
    # self.down1 = nn.Conv2d(nconv, nconv * 2, 3, stride=2, padding=1)
    # self.down2 = nn.Conv2d(nconv * 2, nconv * 4, 3, stride=2, padding=1)

    # self.spec_block = SpectralBlock(nconv * 4, modes=16)

    # self.up1 = nn.ConvTranspose2d(nconv * 4, nconv * 2, 2, stride=2)
    # self.up2 = nn.ConvTranspose2d(nconv * 2, nconv, 2, stride=2)
    # self.outc = nn.Conv2d(nconv, 1, 1)



  def forward(self,x):

    x = self.inc(x)
    x = self.down1(x)
    x = self.down2(x)
    #x = self.down3(x)

    x = self.spec_block(x)
    #x = self.spec_block2(x)

    #x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)

    # x = F.relu(self.inc(x))
    # x = F.relu(self.down1(x))
    # x = F.relu(self.down2(x))

    # x = self.spec_block(x)

    # x = F.relu(self.up1(x))
    # x = F.relu(self.up2(x))

    logits = self.outc(x)

    P0_phase = logits
    P0_phase = torch.tanh(P0_phase) # tanh activation (-1 to 1) 
    P0_phase = P0_phase*np.pi # restore to (-pi, pi) range

    # no_elements_per_side = 11
    # phase_elem = F.interpolate(P0_phase, size=(no_elements_per_side, no_elements_per_side), mode='area')
    # P0_phase = F.interpolate(phase_elem, size=(64, 64), mode='nearest')

    amp = 1
    #Create the complex number
    P0 = torch.complex(amp*torch.cos(P0_phase),amp*torch.sin(P0_phase))

    #==================== Forward Propagation ====================
    dx = Lx/P0.shape[2]
    P_z_ASM = ASM(P0, dx, z)

    # Normalise
    max_vals = torch.amax(torch.abs(P_z_ASM), dim=(2, 3), keepdim=True)
    P_z_ASM = P_z_ASM / max_vals
    P_z_magn = torch.abs(P_z_ASM)

    return P_z_magn, P0_phase
  

# dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/acoustic_DNN/data/acoustic_vortex/"
# images = np.load(dir + "acoustic_traps.npy")[:10]

# images = images[:, np.newaxis ]
# images = torch.Tensor(images)

# model = recon_model()
# P_z_magn, P0_phase = model(images)

