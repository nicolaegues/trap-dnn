

import torch 
from torch import nn
import numpy as np
from numpy.fft import fft2, fftshift

"""
This version has been adapted (restructured, and slightly modified) from the AutoPhase NN model: https://github.com/YudongYao/AutoPhaseNN/tree/main

"""


def ASM(P0,dx,z, k):
    # Zero_padding to hint to fft that field is consistent to infinities


    Nk = 2**int(np.ceil(np.log2(P0.shape[2]))+1)
    kmax = 2*np.pi/dx

    kv = torch.fft.fftfreq(Nk)*kmax # Compute the spatial frequencies
    kx, ky = torch.meshgrid(kv, kv, indexing='ij')
    
    kz = torch.sqrt((k**2 - kx**2 - ky**2).to(torch.complex64))# Allow for complex values
    
    H = torch.exp(-1j*kz*z)

    D = (Nk-1)*dx
    kc = k*torch.sqrt(torch.tensor(0.5*(D**2)/(0.5*D**2 + z**2))) # What is kc and why is it needed instead of just k?
    H[torch.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate

    #P0_padded = 
    P0_fourier = torch.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
    P_z_fourier = P0_fourier * H

    P_z = torch.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field
    P_z = P_z[..., :P0.shape[2], :P0.shape[3]]
    # P_z *= np.exp(-1j*pi/2) # Phase fudge factor to match Huygens

    return P_z



nconv = 64
H = 64


class Conv(nn.Module):
    """(convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Up(nn.Module): 
   
    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.deconv = nn.Sequential(
    
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
    
    def forward(self, x):
        return self.deconv(x)
    
class Out(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            
        )

    def forward(self, x):
        return self.out(x)

class recon_model(nn.Module):

  def __init__(self):
    super(recon_model, self).__init__()
    
    #ENCODER
    # test doing Down instead of inc layer
    # test using same stride
    # test using diff strides for last down like the others.

    self.inc = Conv(1, nconv, stride = 1) 
    self.down1 = Down(nconv, nconv*2)
    self.down2= Down(nconv*2, nconv*4)
    self.down3 = Conv(nconv*4, nconv*8, stride = 2)
    self.down4 = Conv(nconv*8, nconv*8, stride = 2)

    #DECODER
    self.up1 = Up(nconv*8, nconv*4)
    self.up2 = Up(nconv*4,  nconv*2)
    self.up3 = Up(nconv*2,  nconv)
    self.outc = Out(nconv, 1)

    

  def forward(self,x):

    x = self.inc(x)

    x = self.down1(x)
    x = self.down2(x)
    x = self.down3(x)
    x = self.down4(x)

    x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)

    logits = self.outc(x)

    P0_phase = logits
    P0_phase = torch.tanh(P0_phase) # tanh activation (-1 to 1) 
    P0_phase = P0_phase*np.pi # restore to (-pi, pi) range

    amp = 1
   
    #Create the complex number
    P0 = torch.complex(amp*torch.cos(P0_phase),amp*torch.sin(P0_phase))

    ############################################333

    c_w = 1480
    c_p = 2340

    #### wave parameters ####
    f = 1e6
    wavelength = c_w/f
    k = 2*np.pi*f/c_w

    #### Source Parameters ####
    element_width = 3e-3
    kerf = 0.1e-3
    N_elements_per_side = 7
    pitch = element_width + kerf
    aperture = N_elements_per_side*pitch - 2*kerf

    dx = wavelength/4
    z = 1.5*aperture


    P_z_ASM = ASM(P0, dx, z, k)
    P_z_magn = torch.abs(P_z_ASM)

    return P_z_magn, P0_phase
  

# dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/acoustic_DNN/data/acoustic_vortex/"
# images = np.load(dir + "acoustic_traps.npy")[:10]

# images = images[:, np.newaxis ]
# images = torch.Tensor(images)

# model = recon_model()
# P_z_magn, P0_phase = model(images)

