

import numpy as np
from skimage.draw import polygon
from scipy.ndimage import gaussian_filter


# fix: 

#out of frame: position and size are not independent. how to account for this?
    #generate random vals for both. then check with if statement. if (eq to out of frame), then make position such that it's at the limit of what it can be to keep it in frame. ?

#slits at angle not smooth



def square_aperture(img_size, square_size = 4, angle = 0, centre = None):

    if centre is None:
        centre = (img_size // 2, img_size // 2)

    y, x = np.meshgrid(np.arange(img_size), np.arange(img_size))

    x = x - centre[1]
    y = y - centre[0]

    theta = np.deg2rad(-angle)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)

    half = square_size / 2
    mask = (np.abs(x_rot) <= half) & (np.abs(y_rot) <= half)

    return mask


def circular_aperture(img_size, radius = 4, centre = None): 

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = img_size // 2, img_size //2 

    y, x = np.ogrid[:img_size, :img_size]

    dist_from_centre = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    mask = dist_from_centre <= radius

    img[mask] = 1

    return img

def ellipse_aperture(img_size, r1=10, r2=5, angle = 0, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    y, x = np.meshgrid(np.arange(img_size), np.arange(img_size))
    x = x - centre[1]
    y = y - centre[0]

    theta = np.deg2rad(angle)

    # (the 2D roation matrix applied to every (x, y) point)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    mask = (x_rot / r1) ** 2 + (y_rot / r2) ** 2 <= 1
    img[mask] = 1

    return img

def ring_aperture(img_size, r_outer=10, r_inner=5, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    y, x = np.ogrid[:img_size, :img_size]
    dist_from_centre = np.sqrt((x - centre[1])**2 + (y - centre[0])**2)
    mask = (dist_from_centre <= r_outer) & (dist_from_centre >= r_inner)
    img[mask] = 1

    return img

def polygon_aperture(img_size, radius=10, sides=5, angle = 0, centre=None):


    if centre is None:
        centre = (img_size // 2, img_size // 2)

    rotation = np.deg2rad(angle)

    theta = np.linspace(0, 2*np.pi, sides+1) + rotation
    
    x = centre[1] + radius * np.cos(theta)
    y = centre[0] + radius * np.sin(theta)
    
    rr, cc = polygon(y, x)

    img = np.zeros((img_size, img_size))
    img[rr.astype(int) % img_size, cc.astype(int) % img_size] = 1
    return img

def slit_aperture(img_size, slit_width=4, angle=0, centre = None):

    img = np.zeros((img_size, img_size))

    y, x = np.ogrid[:img_size, :img_size]
    if centre is None: 
        centre = img_size // 2, img_size // 2
    
    x = x - centre[0]
    y = y - centre[1]

    theta = np.deg2rad(angle)

    perp_dist = x * np.sin(theta) - y * np.cos(theta)

    mask = np.abs(perp_dist) <= (slit_width / 2)
    img[mask] = 1

    return img

def gaussian_aperture(img_size, sigma=10, centre=None):
    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    img[centre[0], centre[1]] = 1
    img = gaussian_filter(img, sigma=sigma)
    img = img / img.max()  # Normalize to [0,1]
    return img

##################### Aperture Arrays #####################


def slit_array_aperture(img_size, slit_width=4, no_slits = 4, spacing=10, angle=0, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = img_size // 2, img_size // 2

    theta = np.deg2rad(angle)

    ux, uy = -np.sin(theta), np.cos(theta)

    start = -(no_slits-1)/2 * spacing

    for i in range(no_slits):
        offset = start + i*spacing
        cx = int(centre[0] + ux*offset)
        cy = int(centre[1] + uy*offset)
        img += slit_aperture(img_size, slit_width, angle, centre=(cx, cy))

    return img


def circle_array_aperture(img_size, radius=4, spacing=10, grid_shape=(2,2), centre=None):
    
    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    total_w = grid_shape[1] * spacing
    total_h = grid_shape[0] * spacing

    top_left_x = centre[0] - total_h // 2
    top_left_y = centre[1] - total_w // 2

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            x = top_left_x + i * spacing
            y = top_left_y + j * spacing
            img += circular_aperture(img_size, radius, (x, y))

    return img

def square_array_aperture(img_size, square_size = 4, spacing = 10, grid_shape = (2, 2), angle = 0, centre = None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = img_size//2, img_size//2 

    total_w = grid_shape[1] * spacing
    total_h = grid_shape[0] * spacing

    top_left_x = centre[0] - total_h // 2
    top_left_y = centre[1] - total_w // 2

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            x = top_left_x + i * spacing
            y = top_left_y + j * spacing
            img += square_aperture(img_size, square_size, angle, (x, y))

    return img

def ellipse_array_aperture(img_size, r1=10, r2=5, spacing=20, grid_shape=(2,2), angle=0, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    total_w = grid_shape[1] * spacing
    total_h = grid_shape[0] * spacing

    top_left_x = centre[0] - total_h // 2
    top_left_y = centre[1] - total_w // 2

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            x = top_left_x + i * spacing
            y = top_left_y + j * spacing
            img += ellipse_aperture(img_size, r1, r2, angle, (x, y))

    return img

def polygon_array_aperture(img_size, radius=10, sides=6, spacing=20, grid_shape=(2,2), angle=0, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    total_w = grid_shape[1] * spacing
    total_h = grid_shape[0] * spacing

    top_left_x = centre[0] - total_h // 2
    top_left_y = centre[1] - total_w // 2

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            x = top_left_x + i * spacing
            y = top_left_y + j * spacing
            img += polygon_aperture(img_size, radius, sides, angle, (x, y))

    return img

def multi_ring_aperture(img_size, r_inner=5, num_rings=3, ring_thickness=2, spacing=2, centre=None):

    img = np.zeros((img_size, img_size))
    if centre is None:
        centre = (img_size // 2, img_size // 2)

    for i in range(num_rings):
        r_in = r_inner + i * (ring_thickness + spacing)
        r_out = r_in + ring_thickness
        img += ring_aperture(img_size, r_outer=r_out, r_inner=r_in, centre=centre)

    return img

