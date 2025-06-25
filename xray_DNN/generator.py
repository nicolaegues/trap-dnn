
import maxEnt_simulation.DNN.xray_DNN.apertures as ap
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
import os


# To vary: 
# - aperture shape
# - aperture size
# - aperture position (centre) - this changes the phase, but not the diffraction intensity pattern 
# - aperture orientation (angle)
# - in the case of aperture arrays: vary number of apertures


# fix: 

#position and size are not independent. don't want aperture going out of frame. how to account for this?
    #generate random vals for both. then check with if statement. if (eq to out of frame), then make position such that it's at the limit of what it can be to keep it in frame. ?

#overlapping apertures in arrays whne spacing too small rel. to size
#slits at angle not smooth

# generalize for variable img_sizes? 



def diffract(amplitude): 
    """
    Args: 
    real-space amplitude and real-space phase.
    """

    #psi = amplitude * np.exp(1j * phase)
    psi = amplitude

    F = fft2(psi, norm='ortho') # the complex-valued FT of the real-space image. 
    F_shifted = fftshift(F)  #to move the 0 frequency (center of the beam) to the centre of the image - better visualisation.
    
    intensity = np.abs(F_shifted)**2
    rec_phase = np.arctan2(F_shifted.imag, F_shifted.real) # reciprocal space phase. Not measureable at detector. crucial if want to reconsturct the og image by taking inverse FT.

    return intensity, rec_phase

def plot(aperture, intensity, rec_phase):

    fig, axes = plt.subplots(1, 3, figsize = (10, 15))
    axes[0].imshow(aperture)
    axes[0].set_title("Aperture (Amplitude)")
    axes[1].imshow(intensity)
    axes[1].set_title("Diffraction pattern")
    axes[2].imshow(rec_phase)
    axes[2].set_title("Reciprocal Phase")

    for ax in axes: 
        ax.set_axis_off()

    plt.show()

#shape_options = ["square", "circle" ,"ellipse", "ring", "polygon", "slit",  "square array", "circle array", "ellipse array", "multiple rings", "polygon array", "slit array" ]
shape_options = ["square"]

def generate(iterations, dir, img_size = 64): 


    aperture_arr = np.zeros(shape = (iterations, img_size, img_size))
    intensity_arr = np.zeros(shape = (iterations, img_size, img_size))
    phase_arr = np.zeros(shape = (iterations, img_size, img_size))


    for i in range(iterations): 

        rng = np.random.default_rng()

        shape_idx = rng.integers(0, len(shape_options))
        shape = shape_options[shape_idx]
        #print(shape)

        # random centre position generation
        # lower_coord, upper_coord = img_size*0.25, img_size*0.75
        # centre_x = rng.integers(lower_coord, upper_coord)
        # centre_y = rng.integers(lower_coord, upper_coord)
        # centre = centre_x, centre_y
        centre = img_size//2, img_size//2

        lower_angle, upper_angle = 0, 180
        angle = (upper_angle-lower_angle)*rng.random() + lower_angle
        angle = 0

        size =  rng.integers(2, 15)
        size = 6

        #pick spacing after width so it can't violate the constraint
        min_gap = 4
        lower_spacing = size + min_gap
        spacing = rng.integers(size + min_gap, lower_spacing + 10)

        grid_shape_0 = rng.integers(2, 10)
        grid_shape_1 = rng.integers(2, 10)


        if shape == "square": 

            aperture = ap.square_aperture(img_size=img_size, square_size=size, angle = angle, centre = centre)

        elif shape == "circle": 

            aperture = ap.circular_aperture(img_size=img_size,radius = size, centre = centre)
        
        elif shape == "ellipse": 

            r1 = size
            r2 = rng.integers(2, 30)
            aperture = ap.ellipse_aperture(img_size=img_size,r1=r1, r2=r2, angle = angle, centre=centre)
        
        elif shape == "ring": 

            r1 = rng.integers(2, 15)
            r2 = rng.integers(2, 15)

            if r1 < r2: 
                r_outer = r2
                r_inner = r1
            else: 
                r_outer = r1
                r_inner = r2

            aperture = ap.ring_aperture(img_size=img_size,r_outer = r_outer, r_inner=r_inner, centre = centre)

        elif shape == "polygon": 

            sides = rng.integers(2, 10) 
            aperture = ap.polygon_aperture(img_size=img_size,radius=size, sides=sides, angle = angle, centre=centre)

        elif shape == "slit": 

            aperture = ap.slit_aperture(img_size=img_size,slit_width=size, angle=angle, centre = centre)

        elif shape == "gaussian": 

            lower_sigma, upper_sigma = 2, 30
            sigma= (upper_sigma -lower_sigma)*rng.random() + lower_sigma
            aperture = ap.gaussian_aperture(img_size=img_size, sigma=sigma, centre=centre)

        elif shape == "square array": 

            aperture = ap.square_array_aperture(img_size=img_size,square_size = size , spacing  = spacing, grid_shape = (grid_shape_0, grid_shape_1), angle = angle, centre = centre)

        elif shape == "circle array": 

            radius = size/2
            circle_min_gap = 2
            low = size + circle_min_gap
    
            circle_spacing = rng.integers(low, low + 10)

            aperture = ap.circle_array_aperture(img_size=img_size,radius=radius, spacing=circle_spacing, grid_shape=(grid_shape_0, grid_shape_1), centre=centre)

        elif shape == "ellipse array": 
            r1 = size/2
            r2 = rng.integers(2, 15)

            if r1 > r2: 
                r2 = r1
                r1 = r2

            low = r2*2 
            circle_spacing = rng.integers(low, low + 10)

            aperture = ap.ellipse_array_aperture(img_size=img_size,r1=r1, r2=r2, spacing=circle_spacing, grid_shape=(grid_shape_0, grid_shape_1), angle=angle, centre=centre)

        elif shape == "multiple rings": 
            
            r_inner = rng.integers(5, 20)
            ring_thickness = rng.integers(2, 6)
            num_rings = rng.integers(2, 15)

            aperture = ap.multi_ring_aperture(img_size=img_size, r_inner=r_inner, num_rings=num_rings, ring_thickness=ring_thickness, spacing=spacing, centre=centre)

        elif shape == "polygon array": 
            
            radius = size/2
    

            sides = rng.integers(2, 10) 
            aperture = ap.polygon_array_aperture(img_size=img_size,radius=radius, sides=sides, spacing=spacing, grid_shape=(grid_shape_0, grid_shape_1), angle=angle, centre=centre)

        elif shape == "slit array": 

            no_slits = rng.integers(2, 10) 

            aperture = ap.slit_array_aperture( img_size=img_size,slit_width= size, no_slits = no_slits, spacing=spacing, angle=angle, centre=centre)


        aperture = aperture/aperture.max()   

        #deal with situation where aperture is not in frame
        
        if not np.isnan(aperture.max()):
            intensity, rec_phase = diffract(aperture)

            aperture_arr[i] = aperture
            intensity_arr[i] = intensity
            phase_arr[i] = rec_phase

        plot(aperture, intensity, rec_phase)
 
    # np.save(os.path.join(dir, "apertures.npy"), aperture_arr)
    # np.save(os.path.join(dir, "intensities.npy"), intensity_arr)
    # np.save(os.path.join(dir, "rec_phases.npy"), phase_arr)










    






