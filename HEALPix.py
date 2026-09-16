import healpy as hp
import numpy as np
import logging

NSIDE = 32
NPIX = hp.nside2npix(NSIDE)

# Seed vectors — tune these to match your plot
# (az, alt) in degrees → unit vector
def azel_to_vec(az_deg, alt_deg):
    az = np.radians(az_deg)
    alt = np.radians(alt_deg)
    x = np.cos(alt) * np.cos(az)
    y = np.cos(alt) * np.sin(az)
    z = np.sin(alt)
    return np.array([x, y, z])

seeds = np.array([
    azel_to_vec(30, 70), 
    azel_to_vec(330, 70),  
    azel_to_vec(160,  -15),
    azel_to_vec(200,  -15)   
])

# Get unit vectors for all pixels
vecs = np.stack(hp.pix2vec(NSIDE, np.arange(NPIX)), axis=1)  # (NPIX, 3)

# Assign each pixel to nearest seed (max dot product = min angular distance)
dots = vecs @ seeds.T        # (NPIX, 3)
region_map = np.argmax(dots, axis=1)


import matplotlib
import matplotlib.pyplot as plt

hp.mollview(region_map, title="Tiling of Sphere", cmap="Set1", rot=(180,0,0))
hp.graticule()
plt.show()