
import torch 
from torch import nn
import numpy as np
from numpy.fft import fft2, fftshift

"""
This version has been adapted (restructured, and slightly modified) from the AutoPhase NN model: https://github.com/YudongYao/AutoPhaseNN/tree/main

"""


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

    amp= logits
    amp = torch.sigmoid(amp)
    
    # amp_np = amp.detach().numpy()
    # threshold = np.percentile(amp_np, 90)

    # hard = (amp> threshold).float()  
    # amp  = hard + (amp- amp.detach())


    #amp = torch.clip(amp, min=0, max=1.0)
    
    #Apply the support to amplitude
    #mask = torch.tensor([0,1],dtype=amp.dtype, device=amp.device)
    #amp = torch.where(amp<self.sw_thresh,mask[0],amp)
    
    #Restore -pi to pi range
    #ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

    #Pad the predictions to 2X
    # pad = nn.ConstantPad2d(int(H/2),0)
    # amp = pad(amp)
    #ph = pad(ph)

    #Create the complex number
    #complex_x = torch.complex(amp*torch.cos(ph),amp*torch.sin(ph))

    #Compute FT, shift and take abs

    y = torch.fft.fftn(amp ,dim=(-2,-1), norm = "ortho")
    y = torch.fft.fftshift(y,dim=(-2,-1)) #FFT shift will move the wrong dimensions if not specified
    y = torch.abs(y)**2


    
    # #Normalize to scale_I
    # if scale_I>0:
    #     max_I = torch.amax(y, dim=[-1, -2, -3], keepdim=True)
    #     y = scale_I*torch.div(y,max_I+1e-6) #Prevent zero div
    
    # #get support for viz
    # support = torch.zeros(amp.shape,device=amp.device)
    # support = torch.where(amp<self.sw_thresh,mask[0],mask[1])
    #return amp, ph, support

    return y, amp
  
# import numpy as np

# dir = "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/data/"

# images = np.load(dir + "intensities.npy")[:10]
# images = images[:, np.newaxis ]
# images = torch.Tensor(images)

# model = recon_model()
# out = model(images)
