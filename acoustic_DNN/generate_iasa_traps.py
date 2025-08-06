
"""
need to generalise for >2 traps

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from generate_binary_traps import get_random_trap_coords, make_binary_trap_target, make_gaussian_trap_target
from generate_superposed_traps import ASM



#================================== Constants ==================================

# This part below is mainly to get a realistic aperture size
# Source Parameters 
element_width = 3e-3 # Width of one transducer element
kerf = 0.1e-3 # Gap between elements
N_elements_per_side = 7 # Square grid (7x7 elements)
pitch = element_width + kerf # Distance between centers of adjacent elements 
aperture = N_elements_per_side*pitch - 2*kerf # Full aperture width

# size of () (slighlty larger than aperture)
Lx = 1.1*aperture 

# Focal plane distance (propagation depth)
z = 1.5*aperture


#================================== Plotting Functions ==================================

def plot(phase, intensity, dir):

    fig, axes = plt.subplots(1, 2, figsize = (15, 10))
    # pos1 = axes[0].imshow(phase, cmap = "twilight")
    # axes[0].set_title("Phase field")
    pos1 = axes[0].imshow(phase, cmap = "twilight")
    axes[0].set_title("Target")
    pos2 = axes[1].imshow(intensity, cmap = "inferno")
    axes[1].set_title("Acoustic Field Magnitude")

    for ax in axes: 
        ax.set_axis_off()

    fig.colorbar(pos1, ax = axes[0], shrink = 0.55)
    fig.colorbar(pos2, ax = axes[1], shrink = 0.55)
    #plt.savefig(dir)
    plt.show()
    pass

def plot_comp(target, i_0, i_1, i_final, dir): 

    P0_angle_0, Pz_magn_0 = i_0
    P0_angle_1, Pz_magn_1 = i_1
    P0_angle_f, Pz_magn_f = i_final

    fig, axes = plt.subplots(4, 2, figsize = (10, 10))

    pos0 = axes[0][1].imshow(target)
    axes[0][1].set_title("Target")

    pos1 = axes[1][0].imshow(P0_angle_0, cmap = "twilight")
    axes[1][0].set_title("Phase field (iter = 1)")
    pos2 = axes[1][1].imshow(Pz_magn_0)
    axes[1][1].set_title("Acoustic Field Magnitude (iter = 1)")

    pos3 = axes[2][0].imshow(P0_angle_1, cmap = "twilight")
    axes[2][0].set_title("Phase field (iter = 2)")
    pos4 = axes[2][1].imshow(Pz_magn_1)
    axes[2][1].set_title("Acoustic Field Magnitude (iter = 2)")

    pos5 = axes[3][0].imshow(P0_angle_f, cmap = "twilight")
    axes[3][0].set_title("Phase field (iter = 200)")
    pos6 = axes[3][1].imshow(Pz_magn_f)
    axes[3][1].set_title("Acoustic Field Magnitude (iter = 200)")

    for ax in axes:
        for ax_ in ax: 
            ax_.set_axis_off()

    fig.colorbar(pos0, ax = axes[0][1])
    fig.colorbar(pos1, ax = axes[1][0])
    fig.colorbar(pos2, ax = axes[1][1])
    fig.colorbar(pos3, ax = axes[2][0])
    fig.colorbar(pos4, ax = axes[2][1])
    fig.colorbar(pos5, ax = axes[3][0])
    fig.colorbar(pos6, ax = axes[3][1])

    #plt.savefig(dir)
    plt.show()
    pass

#================================== IASA Core Algorithm ==================================

def IASA(target, shape, focal_points, dx, iterations = 200): 
    """
    Iterative Angular Spectrum Algorithm (IASA) to compute a phase-only pattern 
    for trapping particles at given focal points.
    
    The algorithm iteratively enforces constraints in the transducer and focal 
    planes by forward and backward propagating fields using the Angular Spectrum Method.
    
    After a warm-up period (default: 50 iterations), the algorithm evaluates the 
    acoustic intensity at the trap coordinates and adjusts the target field by 
    reweighting the amplitudes to balance trap strength. The new target is used 
    for the remaining iterations to help converge to uniform trap intensity.

    Args:
        target (np.ndarray): Target amplitude distribution in focal plane.
        shape (tuple): Shape of the phase/intensity map.
        focal_points (list): List of (x, y) coordinates for traps.
        dx (float): Pixel spacing in source plane.
        iterations (int): Number of IASA iterations.

    Returns:
        tuple: (Final phase pattern, intermediate result at iter=0, at iter=1)
    """
    # Backward popagte the target to transducer plane 
    P0 = ASM(target, dx, z)
    P0 = np.conj(P0)

    #Phase-only constraint
    phase = np.angle(P0)
    P0 = np.exp(1j*phase)

    for i in range(iterations):

        # Forward propagate
        P_z = ASM(P0, dx, z)

        # Store for visualisation later and print final trap magnitudes
        if i == 0: 
            i_0 = np.angle(P0), np.abs(P_z)
        if i == 1: 
            i_1 = np.angle(P0), np.abs(P_z)

        if i == 199:
            trap_magns = []
            for pt in focal_points:
                magn = np.abs(P_z) 
                trap_magn = magn[pt[1], pt[0]]
                trap_magns.append(trap_magn)
            print(trap_magns)
            print("")

        #============ Applying weights to the traps ============
        if i == 50: # warm-up period until then!

            # Compute field strength at each trap and re-weight
            trap_magns = []
            for pt in focal_points:
                magn = np.abs(P_z) 
                trap_magn = magn[pt[1], pt[0]]
                trap_magns.append(trap_magn)
            print(trap_magns)

            # Compute the factor that dimmer trap needs to be multiplied with to be equal with stronger one. 
            if trap_magns[0] < trap_magns[1]: 
                factor = trap_magns[1]/trap_magns[0]/1.1
                weights = [factor, 1]
            else: 
                factor = trap_magns[0]/trap_magns[1]/1.1
                weights = [1, factor]

             # Rebuild weighted target
            target = make_gaussian_trap_target(shape, focal_points, weights = weights)
            #target = make_binary_trap_target(shape, focal_points, weights = weights)

        P_z_angle = np.angle(P_z)
        P_z_sugg =  target* np.exp(1j*P_z_angle)

        #backpropagate
        P0 = ASM(P_z_sugg, dx, z)
        P0 = np.conj(P0)

        #Phase-only constraint
        phase = np.angle(P0)
        P0 = np.exp(1j*phase)

    return P0, i_0, i_1

#================================== Data Generator ==================================
def generate_IASA(no_samples, output_dir, shape=(64, 64), n_traps=2):
    """
    Generates synthetic acoustic phase data using IASA.

    Args:
        no_samples (int): Number of samples to generate.
        output_dir (str): Directory to save .npy files.
        shape (tuple): Shape of output arrays.
        n_traps (int): Number of acoustic traps per target.
    """
    height, width = shape
    dx = Lx / width

    phases = np.zeros((no_samples, *shape))
    traps = np.zeros_like(phases)
    coords = []

    for i in range(no_samples):

        # randomly place n_traps
        focal_points = get_random_trap_coords(n_traps)

        target = make_gaussian_trap_target(shape, focal_points)
        #target = make_binary_trap_target(shape, focal_points)

        P0, i_0, i_1 = IASA(target, shape, focal_points, dx)
        pressure = ASM(P0, dx, z)
        P0_phase = np.angle(P0)
        i_final = P0_phase, np.abs(pressure)

        #evaluate(i_final, focal_points)

        #plot(P0_phase, i_final[1], f"{output_dir}{i}")
        #plot_comp(target, i_0, i_1, i_final, f"{output_dir}{i}")

        phases[i] = P0_phase
        traps[i] = np.abs(pressure)
        coords.append(focal_points)


    np.save(os.path.join(output_dir, "acoustic_phases.npy"), phases)
    np.save(os.path.join(output_dir, "acoustic_traps.npy"), traps)
    np.save(os.path.join(output_dir, "trap_coords.npy"), np.array(coords))


if __name__ == "__main__":   

    dir = os.getcwd() 
    print(dir)
    train_dir =  dir + "/iasa_figs/testing_off_axis/try_3/"
    # test_dir =  dir + "/data/test/"

    generate_IASA(20, train_dir)



