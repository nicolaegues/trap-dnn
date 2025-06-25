
""" Main code belongs to Barney Emmens """


import numpy as np
import matplotlib.pyplot as plt
import os

c_w = 1480
c_p = 2340 # speed of sound in PS https://ims.evidentscientific.com/en/learn/ndt-tutorials/thickness-gauge/appendices-velocities

#### Define wave Parameters ####
f = 1e6
wavelength = c_w/f
k = 2*np.pi*f/c_w

#### Source Parameters ####
element_width = 3e-3
kerf = 0.1e-3
N_elements_per_side = 7
pitch = element_width + kerf
aperture = N_elements_per_side*pitch - 2*kerf


def plot(phase, intensity):

    fig, axes = plt.subplots(1, 2, figsize = (10, 15))
    axes[0].imshow(phase)
    axes[0].set_title("Phase field")
    axes[1].imshow(intensity)
    axes[1].set_title("Diffraction pattern")

    for ax in axes: 
        ax.set_axis_off()

    plt.show()


def phaseField(X,Y,focalPoint,m):
    
    #Translated the coordinate system: each grid point(X, Y) is shifted to  align with the desired focus point. 
    Xf = X - focalPoint[0]
    Yf = Y - focalPoint[1]
    Zf = focalPoint[2]
    
    r = np.sqrt(Xf**2 + Yf**2 + Zf**2) # euclidean distance from each source point to the focus
    phi = np.arctan2(Yf, Xf) 

    out = np.exp(1j*phi*m)*np.exp(1j*k*r) # Vortex of order m
    
    return out

def phaseFieldTwin(X, Y, focalPoint1, focalPoint2, m):

    #Translated coordinates for focal point 1
    Xf1 = X - focalPoint1[0]
    Yf1 = Y - focalPoint1[1]
    Zf1 = focalPoint1[2]
    r1 = np.sqrt(Xf1**2 + Yf1**2 + Zf1**2)
    phi1 = np.arctan2(Yf1, Xf1)
    field1 = np.exp(1j * phi1 * m) * np.exp(1j * k * r1)

    #Translated coordinates for focal point 2
    Xf2 = X - focalPoint2[0]
    Yf2 = Y - focalPoint2[1]
    Zf2 = focalPoint2[2]
    r2 = np.sqrt(Xf2**2 + Yf2**2 + Zf2**2)
    phi2 = np.arctan2(Yf2, Xf2)
    field2 = np.exp(1j * phi2 * m) * np.exp(1j * k * r2)

    #Combine both fields
    out = field1 + field2

    return out

def phaseFieldRandomTraps(X, Y, n_traps=1, max_order=0):

    field = np.zeros_like(X, dtype=complex)
    rng = np.random.default_rng()

    for _ in range(n_traps):

        x_fp = rng.uniform(-aperture/2, aperture/2)
        y_fp = rng.uniform(-aperture/2, aperture/2)
        #z_fp = rng.uniform(1.2*aperture, 1.8*aperture) 
        z_fp = 1.5*aperture

        focalPoint = [x_fp, y_fp, z_fp]
        m = rng.integers(-max_order, max_order + 1)

        field += phaseField(X, Y, focalPoint, m)

    return field

def genP0(focalPoint, focalPoint2, m, dx, Lx, n_traps):

    xv = np.arange(-Lx/2, Lx/2, dx)
    yv = np.arange(-Lx/2, Lx/2, dx)
    X, Y = np.meshgrid(xv, yv)
    P0 = np.ones(X.shape, dtype=complex)

    # Make square source - side length = aperture
    P0[X>aperture/2] = 0;
    P0[Y>aperture/2] = 0;
    P0[X<-aperture/2] = 0;
    P0[Y<-aperture/2] = 0;

    # apply a spiral phase term that generates a vortex centred around a focalpoint.
    #HDPhases = phaseField(X, Y, focalPoint, m) 

    #HDPhases = phaseFieldTwin(X, Y, focalPoint,focalPoint2, m) 

    HDPhases = phaseFieldRandomTraps(X, Y, n_traps=n_traps, max_order=0)

    P0 = P0*HDPhases

    return P0,xv,yv

def ASM(P0,dx,z):
    # Zero_padding to hint to fft that field is consistent to infinities

    test = P0.shape[0]
    Nk = 2**int(np.ceil(np.log2(P0.shape[0]))+1)
    kmax = 2*np.pi/dx

    kv = np.fft.fftfreq(Nk)*kmax # Compute the spatial frequencies
    kx, ky = np.meshgrid(kv, kv)
    kz =  np.emath.sqrt(k**2 - kx**2 - ky**2) # Allow for complex values
    
    # H = np.exp(-1j*kz*z)/kz
    H = np.exp(-1j*kz*z)

    D = (Nk-1)*dx
    kc = k*np.sqrt(0.5*(D**2)/(0.5*D**2 + z**2)) # What is kc and why is it needed instead of just k?
    H[np.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate

    P0_fourier = np.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
    P_z_fourier = P0_fourier * H

    P_z = np.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field
    P_z = P_z[:P0.shape[0],:P0.shape[1]]
    # P_z *= np.exp(-1j*pi/2) # Phase fudge factor to match Huygens

    return P_z



m = 0
dx = wavelength/4
Lx = 1.1*aperture


############################################################

def generate(no_samples, dir): 

    shape = (no_samples, int(np.ceil(Lx/dx)), int(np.ceil(Lx/dx)))
    phases_array = np.zeros(shape)
    trap_array = np.zeros(shape)

    # I want my train set's focal point to be within the upper diagonal, and my test set within the lower. 

    upper = 0.5
    lower = -0.5
    rng = np.random.default_rng()

    for i in range(no_samples):

        # while True: 
        #     x_factor = (upper-lower)*rng.random() + lower
        #     y_factor = (upper-lower)*rng.random() + lower
        #     #if y_factor >= - x_factor: 
        #     if y_factor < - x_factor: 
        #         break

        # focalPoint = [x_factor*aperture,-y_factor*aperture,1.5*aperture]

        P0,xv,yv = genP0(0, 0, m, dx, Lx, n_traps=3)


        P0_phase = np.angle(P0)
        X, Y = np.meshgrid(xv, yv)
        P0_phase[X>aperture/2] = 0
        P0_phase[Y>aperture/2] = 0
        P0_phase[X<-aperture/2] = 0
        P0_phase[Y<-aperture/2] = 0

        z = 1.5*aperture
        P_z_ASM = ASM(P0,dx,z)
        P_z_magn = np.abs(P_z_ASM)
        

        phases_array[i] = P0_phase
        trap_array[i] = P_z_magn

        #plot(P0_phase, P_z_magn)

    np.save(os.path.join(dir, "acoustic_phases.npy"), phases_array)
    np.save(os.path.join(dir, "acoustic_traps.npy"), trap_array)


def generate_overfit(no_samples, dir): 
    shape = (no_samples, int(np.ceil(Lx/dx)), int(np.ceil(Lx/dx)))
    phases_array = np.zeros(shape)
    trap_array = np.zeros(shape)

    x_factor = 0.25
    y_factor = 0

    #origin in centre of square. this so that it visually makes sense
    focalPoint = [x_factor*aperture,-y_factor*aperture,1.5*aperture]

    x_factor = -0.25
    y_factor = 0

    #origin in centre of square. this so that it visually makes sense
    focalPoint2 = [x_factor*aperture,-y_factor*aperture,1.5*aperture]


    P0,xv,yv = genP0(focalPoint, focalPoint2, m, dx, Lx)

    P0_phase = np.angle(P0)
    X, Y = np.meshgrid(xv, yv)
    P0_phase[X>aperture/2] = 0
    P0_phase[Y>aperture/2] = 0
    P0_phase[X<-aperture/2] = 0
    P0_phase[Y<-aperture/2] = 0

    z = 1.5*aperture
    P_z_ASM = ASM(P0,dx,z)
    P_z_magn = np.abs(P_z_ASM)
        
    for i in range(no_samples):

        phases_array[i] = P0_phase
        trap_array[i] = P_z_magn

        #plot(P0_phase, P_z_magn)

    np.save(os.path.join(dir, "acoustic_phases.npy"), phases_array)
    np.save(os.path.join(dir, "acoustic_traps.npy"), trap_array)



train_dir =  "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/acoustic_DNN/data/random/train/"
test_dir =  "C:/Users/nicol/OneDrive - University of Bristol/MSc_project-DESKTOP-M3M0RRL/maxEnt_simulation/DNN/acoustic_DNN/data/random/test/"

generate(200, test_dir)
