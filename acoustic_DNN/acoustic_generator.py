
""" Main code belongs to Barney Emmens """


""" Currently just simulating a square-shaped continuous source field 
(total active region has size equal to the aperture), not a discrete one 
(an array of transducer elememts). """


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
pitch = element_width + kerf # = 3.0e-3 + 0.1e-3 = 3.1 mm
aperture = N_elements_per_side*pitch - 2*kerf # ca. 21.3 mm

half = (N_elements_per_side - 1)/2
element_centres = pitch * (np.arange(N_elements_per_side) - half)

m = 0
dx = wavelength/4 #0.37 mm
Lx = 1.1*aperture # 23.43 mm

z = 1.5*aperture

# later define xv = np.arange(-Lx/2, Lx/2, dx). so gives 64x64 as np.ceil(23.43/0.37) = 64


def plot(phase, intensity, coords):

    fig, axes = plt.subplots(1, 2, figsize = (10, 15))
    pos1 = axes[0].imshow(phase)
    axes[0].set_title("Phase field")
    pos2 = axes[1].imshow(intensity, cmap = "inferno")
    axes[1].set_title("Diffraction pattern")
    axes[1].plot(coords[0][0], coords[0][1], "ro")
    axes[1].plot(coords[1][0], coords[1][1], "ro")

    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(pos1, ax = axes[0], shrink = 0.55)
    fig.colorbar(pos2, ax = axes[1], shrink = 0.55)

    plt.show()


def phaseField(X,Y,focalPoint,m):
    
    #Translate the coordinate system: each grid point(X, Y) is shifted to  align with the desired focus point. 
    Xf = X - focalPoint[0]
    Yf = Y - focalPoint[1]
    Zf = focalPoint[2]
    
    r = np.sqrt(Xf**2 + Yf**2 + Zf**2) # euclidean distance from each source point to the focus
    phi = np.arctan2(Yf, Xf) 

    out = np.exp(1j*phi*m)*np.exp(1j*k*r) 
    
    return out

def phaseField_twin(X, Y, focalPoint, m):
    # shift coords so center of vortex is at focalPoint
    Xf = X - focalPoint[0]
    Yf = Y - focalPoint[1]
    Zf = focalPoint[2]
    
    # base vortex + propagation phase
    r   = np.sqrt(Xf**2 + Yf**2 + Zf**2)
    phi = np.arctan2(Yf, Xf)
    base = np.exp(1j*(phi*m + k*r))
    
    # create mask for "right" half (Xf>0)
    
    mask = (Xf > 0)
    # apply π‐step there (i.e. multiply by -1)
    out = base * np.where(mask, -1.0, 1.0)
    
    return out

def phaseField_multiple(X, Y, focalPoints, m):

    field = np.zeros_like(X, dtype=complex)


    #Translated coordinates for focal point 1
    for focalPoint in focalPoints:
        Xf1 = X - focalPoint[0]
        Yf1 = Y - focalPoint[1]
        Zf1 = focalPoint[2]
        r1 = np.sqrt(Xf1**2 + Yf1**2 + Zf1**2)
        phi1 = np.arctan2(Yf1, Xf1)
        field1 = np.exp(1j * phi1 * m) * np.exp(1j * k * r1)

        field += field1

    phase   = np.angle(field)                        
    field = np.exp(1j*phase)

    return field

def phaseFieldRandomTraps(X, Y, n_traps=2, max_order=0):

    field = np.zeros_like(X, dtype=complex)
    rng = np.random.default_rng()

    #n_traps = rng.integers(1, 3)
    coords = []

    for i in range(n_traps):

        x_fp = rng.uniform(-aperture/2, aperture/2)
        y_fp = rng.uniform(-aperture/2, aperture/2)
        z_fp = 1.5*aperture
        # 0.0010312053783934506
        # -0.003165814107889859
        
        focalPoint = [x_fp, y_fp, z_fp]
        m = rng.integers(-max_order, max_order + 1)

        field += phaseField(X, Y, focalPoint, m)

        x_pix = int(np.round((y_fp + Lx/2) / dx)) 
        y_pix = int(np.round((x_fp + Lx/2) / dx)) 
        coords.append([y_pix, x_pix])


    phase   = np.angle(field)                        
    field = np.exp(1j*phase)
    return field, coords

def genP0(focalPoint, m, dx, Lx, n_traps):

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
    #HDPhases = phaseField_multiple(X, Y, focalPoint, m) 
    #HDPhases = phaseField_twin(X, Y, focalPoint, m)
    HDPhases, coords = phaseFieldRandomTraps(X, Y, n_traps=n_traps, max_order=0)

    P0 = P0*HDPhases

    return P0,xv,yv, coords
 

def genP0_discrete(focalPoint, N_elements_per_side, m, dx, Lx):
    xv = np.arange(-Lx/2, Lx/2, dx)
    yv = np.arange(-Lx/2, Lx/2, dx)
    X, Y = np.meshgrid(xv, yv)
    P0 = np.zeros(X.shape, dtype=complex)

    for x in range(N_elements_per_side):
        for y in range(N_elements_per_side):
            xc, yc = element_centres[x], element_centres[y]

            HDPhases = phaseField(xc, yc, focalPoint,m)
            
            mask = (np.abs(X - xc) <= element_width/2) & \
                   (np.abs(Y - yc) <= element_width/2)

            P0[mask] += HDPhases # summed pressure field from each element

    return P0, xv, yv

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


############################################################

def generate(no_samples, dir): 

    shape = (no_samples, int(np.ceil(Lx/dx)), int(np.ceil(Lx/dx)))
    phases_array = np.zeros(shape)
    trap_array = np.zeros(shape)
    coords = []

    # I want my train set's focal point to be within the upper diagonal, and my test set within the lower. 
    for i in range(no_samples):

        # focalPoint = [x_factor*aperture,-y_factor*aperture,1.5*aperture]
        #focalPoint = [[-aperture/4,0,1.5*aperture], [aperture/4,0,1.5*aperture]]
        #focalPoint = [0, 0, 1.5*aperture]
        P0,xv,yv, temp_coords = genP0(0, m, dx, Lx, n_traps=2)


        P0_phase = np.angle(P0)
        X, Y = np.meshgrid(xv, yv)
        P0_phase[X>aperture/2] = 0
        P0_phase[Y>aperture/2] = 0
        P0_phase[X<-aperture/2] = 0
        P0_phase[Y<-aperture/2] = 0

        P_z_ASM = ASM(P0,dx,z)
        P_z_magn = np.abs(P_z_ASM)
        
        phases_array[i] = P0_phase
        trap_array[i] = P_z_magn
        coords.append(temp_coords)
        #print(temp_coords)
        #plot(P0_phase, P_z_magn, temp_coords)

    np.save(os.path.join(dir, "acoustic_phases.npy"), phases_array)
    np.save(os.path.join(dir, "acoustic_traps.npy"), trap_array)
    np.save(os.path.join(dir, "trap_coords.npy"), np.array(coords))


def generate_overfit(no_samples, dir): 
    shape = (no_samples, int(np.ceil(Lx/dx)), int(np.ceil(Lx/dx)))
    phases_array = np.zeros(shape)
    trap_array = np.zeros(shape)

    x_factor = 0
    y_factor = 0

    #origin in centre of square. this so that it visually makes sense
    focalPoint = [x_factor*aperture,-y_factor*aperture,1.5*aperture]

    x_factor = -0.25
    y_factor = 0

    #origin in centre of square. this so that it visually makes sense
    focalPoint2 = [x_factor*aperture,-y_factor*aperture,1.5*aperture]


    P0,xv,yv = genP0(focalPoint, focalPoint2, m, dx, Lx, 0)

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


dir = os.getcwd() 
train_dir =  dir + "/data/random/test/"
test_dir =  dir + "/data/random/test/"

generate(200, test_dir)
