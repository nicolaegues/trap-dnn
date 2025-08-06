
import numpy as np
import os
import re
from utils import plot_4_ims
import cv2
from data_generation.generate_superposed_traps import ASM

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

# desired resolution of final acoustic field
shape = (64, 64)

Lx = 1.1*aperture
dx = Lx/shape[0]

#================================== Experiment selection ==================================
#exp = "KEEP_exp_Thu-03-Jul-2025-at-03-53-57PM"
exp = "z_IASA_exp_Tue-05-Aug-2025-at-01-50-16PM"

#dir = os.getcwd() + "/maxEnt_simulation/DNN/"
dir = os.getcwd()
exp_dir = f"{dir}/experiments/{exp}/"
final_figs_dir = f"{exp_dir}final_figs/"
test_figs_dir = f"{exp_dir}test_figs/"
#select the folder in which to store the reduced-size 
figs_dir = test_figs_dir
reduced_size_dir = f"{figs_dir}{N_elements_per_side}x{N_elements_per_side}/"
os.makedirs(reduced_size_dir, exist_ok=True)

#================================== File Extraction ==================================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

files = sorted(os.listdir(figs_dir), key=natural_sort_key)
np_files = []
coord_files = []
for file in files: 
    coords_match = re.search(r"coords", file)
    match = re.search(r".npy", file)
    if match and not coords_match: 
        np_files.append(file)
    if coords_match: 
        coord_files.append(file)

#================================== Discretisation and Forward-Propagation ==================================

for i, (file, coords_f) in enumerate(zip(np_files, coord_files)): 

    arr = np.load(f"{figs_dir}{file}")
    coords = np.load(f"{figs_dir}{coords_f}")
    og_diffr, P0_phase = arr[0], arr[-1]

    # Downsample to given size 
    phase_elem = cv2.resize(P0_phase, (N_elements_per_side, N_elements_per_side), interpolation=cv2.INTER_AREA)
    #Upsample (Nearest-neighbor interpolation) to get high-resolution discrete"blocks", which can then be propagated forward using ASM
    P0_phase = cv2.resize(phase_elem, (64, 64), interpolation=cv2.INTER_NEAREST)

    #Forward Propagation using ASM
    amp = 1
    # Create the complex number
    P0 = amp * np.exp(1j * P0_phase)

    P_z_ASM = ASM(P0, dx, z)
    P_z_magn = np.abs(P_z_ASM)


    plot_4_ims(og_diffr, P_z_magn, 0, P0_phase, coords, dir = f"{reduced_size_dir}{N_elements_per_side}x{N_elements_per_side}_{i}")
    arr = np.array([P_z_magn, P0_phase])
    np.save(f"{reduced_size_dir}{N_elements_per_side}x{N_elements_per_side}_{i}", arr)




