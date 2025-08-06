
""" 
The Angular Spectrum Method (ASM) function as well as the function to generate the helical phasefield belong to Barney Emmens.
Author of the rest of the script: Nicola Egües
"""


import numpy as np
import matplotlib.pyplot as plt
import os

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

half = (N_elements_per_side - 1)/2
element_centres = pitch * (np.arange(N_elements_per_side) - half)

#size of () (slighlty larger than aperture)
Lx = 1.1*aperture 

# Focal plane distance (propagation depth)
z = 1.5*aperture

m = 0 #topoligal charge of the vortex field (not varied here)


#================================== Main Functions ==================================

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
    Nk = 2**int(np.ceil(np.log2(P0.shape[0]))+1)
    kmax = 2*np.pi/dx

    # Compute spatial frequency grids
    kv = np.fft.fftfreq(Nk)*kmax # Compute the spatial frequencies
    kx, ky = np.meshgrid(kv, kv)
    kz =  np.emath.sqrt(k**2 - kx**2 - ky**2) # Allow for complex values
    
    # Transfer function
    H = np.exp(-1j*kz*z)

    # Limit angular spectrum to propagating waves only
    D = (Nk-1)*dx
    kc = k*np.sqrt(0.5*(D**2)/(0.5*D**2 + z**2)) # Angular cutoff
    H[np.sqrt(kx**2 + ky**2) > kc] = 0 # Wavelengths greater than kc cannot propogate - Zero out evanescent components

    # Propagate the field
    P0_fourier = np.fft.fft2(P0,[Nk,Nk]) # Compute the 2D Fourier Transform of the input field
    P_z_fourier = P0_fourier * H
    P_z = np.fft.ifft2(P_z_fourier,[Nk,Nk]) # Compute the inverse 2D Fourier Transform of the field

    P_z = P_z[:P0.shape[0],:P0.shape[1]]

    return P_z


def plot(phase, intensity, coords):
    """
    Visualise the input phase and resulting intensity pattern.
    """

    fig, axes = plt.subplots(1, 2, figsize = (15, 10))
    pos1 = axes[0].imshow(phase,cmap = "twilight")
    axes[0].set_title("Phase field")
    pos2 = axes[1].imshow(intensity)
    axes[1].set_title("Diffraction pattern")
    axes[1].plot(coords[0][0], coords[0][1], "ro", markersize = 2)
    axes[1].plot(coords[1][0], coords[1][1], "ro", markersize = 2)

    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(pos1, ax = axes[0], shrink = 0.55)
    fig.colorbar(pos2, ax = axes[1], shrink = 0.55)

    plt.show()

def two_traps_moving_closer(no_samples):
    """
    Creates a series of 2-trap coordinate pairs that gradually move closer  along the x-axis.

    Args:
        no_samples (int): Number of samples to generate.

    Returns:
        list: List of the two coordinate pairs [(x1, y1), (x2, y2)] per sample.
    """
    all_coords = []
    for i in range(no_samples): 
        coords = [(4+i, 32), (60-i, 32)]
        all_coords.append(coords)

    return all_coords

def phaseField(X,Y,focalPoint):
    """
    Generates the complex phase field at the source for a single acoustic trap 
    focused at the given 3D point.

    Args:
        X (ndarray): 2D grid of x-coordinates [m].
        Y (ndarray): 2D grid of y-coordinates [m].
        focalPoint (list): Coordinates [x, y, z] of the trap center [m].

    Returns:
        ndarray: Complex-valued phase field.
    """
    
    #Translate the coordinate system: each grid point(X, Y) is shifted to  align with the desired focus point. 
    Xf = X - focalPoint[0]
    Yf = Y - focalPoint[1]
    Zf = focalPoint[2]
    
    r = np.sqrt(Xf**2 + Yf**2 + Zf**2) # euclidean distance from each source point to the focus
    phi = np.arctan2(Yf, Xf) 

    out = np.exp(1j*phi*m)*np.exp(1j*k*r) 
    
    return out


def phaseField_from_coords(X, Y, dx, trap_coords):
    """
    Generates a complex phase field by superposing multiple spherical wavefronts,
    each focused at a trap location specified in pixel coordinates.

    Args:
        X (ndarray): 2D grid of x-coordinates [m].
        Y (ndarray): 2D grid of y-coordinates [m].
        dx (float): Spatial resolution [m] of the grid.
        n_traps (int): Number of traps to randomly place in the aperture.

    Returns:
        ndarray: 2D complex-valued phase field.
    
    """

    field = np.zeros_like(X, dtype=complex)

    for i, j in trap_coords:
        #translate the pixel coordinates
        x = (i + 0.5) * dx - Lx / 2
        y = (j + 0.5) * dx - Lx / 2
        focalPoint = [x, y, z]

        field += phaseField(X, Y, focalPoint)

    # Phase-only constraint
    phase   = np.angle(field)                        
    field = np.exp(1j*phase)

    return field

def phaseFieldRandomTraps(X, Y, dx, n_traps=2):
    """
    Generates a complex phase field by superposing multiple spherical wavefronts,
    each focused at a randomly generated trap location.

    Args:
        X (ndarray): 2D grid of x-coordinates [m].
        Y (ndarray): 2D grid of y-coordinates [m].
        dx (float): Spatial resolution [m] of the grid.
        n_traps (int): Number of traps to randomly place in the aperture.

    Returns:
        ndarray: 2D complex-valued phase field.
        coords: list of tuples: Pixel coordinates [(i, j), ...] of the generated trap positions.
    
    """

    field = np.zeros_like(X, dtype=complex)
    rng = np.random.default_rng(seed = 42)

    coords = []
    for i in range(n_traps):

        x_fp = rng.uniform(-aperture/2, aperture/2)
        y_fp = rng.uniform(-aperture/2, aperture/2)
        z_fp = z
        focalPoint = [x_fp, y_fp, z_fp]

        field += phaseField(X, Y, focalPoint)

        #translate to pixel coords to store
        x_pix = int(np.round((y_fp + Lx/2) / dx)) 
        y_pix = int(np.round((x_fp + Lx/2) / dx)) 
        coords.append((y_pix, x_pix))

    # Phase-only constraint
    phase   = np.angle(field)                        
    field = np.exp(1j*phase)
    return field, coords


def phaseFieldDiscrete_from_coords(X, Y, dx, trap_coords):
    """
    Not in use currently. 

    Simulates the phase field generated by a discrete square transducer array
    focused at one or more focal points. Each transducer element emits a local 
    phase-shifted wave based on its position and the trap location.

    Parameters:
        X (ndarray): 2D grid of x coordinates [m] (source plane).
        Y (ndarray): 2D grid of y coordinates [m] (source plane).
        trap_coords (list of tuples): List of trap pixel coordinates [(i, j), ...].
        dx (float): Grid spacing [m].

    Returns:
        ndarray: 2D complex phase field from the discrete element superposition.
    """

    field = np.zeros_like(X, dtype=complex)

    for i, j in trap_coords: 
        # Convert pixel to real trap coordinates
        x_fp = (i + 0.5) * dx - Lx / 2
        y_fp = (j + 0.5) * dx - Lx / 2
        focalPoint = [x_fp, y_fp, z]

        for xc in element_centres:
            for yc in element_centres:

                HDPhases = phaseField(xc, yc, focalPoint)
                
                mask = (np.abs(X - xc) <= element_width/2) & \
                    (np.abs(Y - yc) <= element_width/2)
                
                field[mask] += HDPhases

    # Phase-only constraint
    phase = np.angle(field)
    return np.exp(1j * phase)


def generate_traps(no_samples, output_dir, shape = (64, 64), trap_coords = None): 
    """
    Generates a dataset of acoustic phase maps and their 
    corresponding propagated intensity (trap) patterns.

    The generated data includes:
        - Phase maps (source plane)
        - Intensity patterns (target plane via ASM)
        - Trap coordinates of the traps in each sample

    Parameters:
        no_samples (int): Number of samples to generate.
        output_dir (str): Path to the folder where .npy output files will be saved.
        shape (tuple of int): Output resolution (height, width) in pixels.
        trap_coords (list or None): List of pixel coordinates per sample.
                                    If None, traps are generated randomly.
    """

    height, width = shape
    dx = Lx / width
    xv = np.arange(-Lx/2, Lx/2, dx)
    yv = np.arange(-Lx/2, Lx/2, dx)
    X, Y = np.meshgrid(xv, yv)

    phases_array = np.zeros(shape = (no_samples, height, width))
    trap_array = np.zeros(shape = (no_samples, height, width))
    all_coords = []

    # I want my train set's focal point to be within the upper diagonal, and my test set within the lower. 
    for i in range(no_samples):

        if trap_coords == None: 
            P0, coords = phaseFieldRandomTraps(X, Y, dx, n_traps=2)
        else: 
            coords = trap_coords[i]
            P0 = phaseField_from_coords(X, Y, dx, coords)

        P0_phase = np.angle(P0)
        #P0_phase = apply_aperture_mask(P0_phase, X, Y)

        P_z_ASM = ASM(P0,dx,z)
        P_z_magn = np.abs(P_z_ASM)
        
        phases_array[i] = P0_phase
        trap_array[i] = P_z_magn
        all_coords.append(coords)
        #print(coords)
        plot(P0_phase, P_z_magn, coords)

    # np.save(os.path.join(output_dir, "acoustic_phases.npy"), phases_array)
    # np.save(os.path.join(output_dir, "acoustic_traps.npy"), trap_array)
    # np.save(os.path.join(output_dir, "trap_coords.npy"), np.array(coords))

if __name__ == "__main__":
    dir = os.getcwd() 
    train_dir =  dir + "/data/random/train/"
    test_dir =  dir + "/data/og_traps_moving/"

    trap_coords = two_traps_moving_closer(20)
    generate_traps(2, output_dir = test_dir, shape = (64, 64), trap_coords=trap_coords)

