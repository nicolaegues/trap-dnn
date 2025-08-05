import numpy as np
import matplotlib.pyplot as plt
import os


def make_binary_trap_target(shape, trap_coords, radius_pixels=2.5):
    """
    Creates a binary target map with circular trap regions set to 1.
    Args:
        shape (tuple): Shape of the 2D output array (height, width).
        trap_coords (list of tuples): List of the (x, y) trap centers.
        radius_pixels (float): Radius of circular trap area in pixels.

    Returns:
        ndarray: Binary target image.
    """
    masks = []
    target = np.zeros(shape, dtype=np.float32)
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    for (cx, cy) in trap_coords:
        mask = (X - cx)**2 + (Y - cy)**2 <= radius_pixels**2
        target[mask] = 1
        masks.append(mask)

    return target

def make_gaussian_trap_target(shape, trap_coords, sigma_px=2.5):
    """
    Creates a target map with Gaussian-shaped trap intensities.
    Args:
        shape (tuple): Shape of the 2D output array (height, width).
        trap_coords (list of tuples): List of the (x, y) trap centers.
        sigma_px (float): Standard deviation of the Gaussian in pixels.

    Returns:
        ndarray: Gaussian target image.
    """

    target = np.zeros(shape, dtype=np.float32)
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for (cx, cy) in trap_coords:
        gauss = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma_px**2))
        target += gauss

    return target


def get_random_trap_coords(n_traps): 
    """
    Randomly generates non-overlapping trap coordinates within a 64x64 grid.
    Ensures traps are spaced apart by at least 12 pixels.

    Args:
        n_traps (int): Number of traps to generate.

    Returns:
        list of tuples: Trap coordinates.
    """

    rng = np.random.default_rng()
    trap_coords = []

    too_near = True
    for t in range(n_traps): 
        fp = (rng.integers(10, 54), rng.integers(10, 54))

        if t != 0:
            while too_near == True: 
                for e_fp in trap_coords: 
                    if np.sqrt(np.sum((np.array(e_fp) - np.array(fp))**2)) >= 12: 
                        too_near = False
                    else: 
                        fp = (rng.integers(10, 54), rng.integers(10, 54))

        trap_coords.append(fp)
    
    return trap_coords


def plot(target): 
    fig, ax = plt.subplots(figsize = (10, 6))
    pos1 = ax.imshow(target)

    ax.set_axis_off()
    fig.colorbar(pos1, ax = ax)
    plt.show()
    

def generate_traps(no_samples, output_dir, shape=(64, 64), n_traps=2, trap_coords = None):
    """
    Generates and saves synthetic trap images.

    Args:
        no_samples (int): Number of samples to generate.
        output_dir (str): Path to directory where the .npy files will be saved.
        shape (tuple): Shape of each 2D trap image.
        n_traps (int): Number of traps per image (used if trap_coords is None).
        trap_coords (list of list of tuples): Optional fixed trap coordinates per sample.
    """
    traps = np.zeros((no_samples, *shape))
    coords = []

    for i in range(no_samples):
        
        # randomly place n_traps if no coordinates are given
        if trap_coords == None: 
            trap_coords = get_random_trap_coords(n_traps)

        #target = make_gaussian_trap_target(shape, trap_coords[i])
        target = make_binary_trap_target(shape, trap_coords[i])

        #plot(target)

        traps[i] = target
        coords.append(trap_coords[i])


    np.save(os.path.join(output_dir, "acoustic_traps.npy"), traps)
    np.save(os.path.join(output_dir, "trap_coords.npy"), np.array(coords))


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


if __name__ == "__main__":
    dir = os.getcwd() 
    train_dir = dir + "/data/binary_traps_moving_closer/"
    #test_dir =  dir + "/data/perfect_binary_traps/test/"

    trap_coords = two_traps_moving_closer(20)

    #y axis 0-->+64: top to bottom. 
    #x axis 0-->+64: left to right
    #trap_coords = [[(4, 32), (60, 8)], [(32, 32), (48, 8)]] #2 traps in each of the two samples

    generate_traps(no_samples=20, output_dir = train_dir, shape = (64, 64), trap_coords = trap_coords)


