import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# 1. Define resolution (keep NSIDE small like 4 or 8 to see distinct tiles clearly)
NSIDE = 1
npix = hp.nside2npix(NSIDE)

# 2. Populate the entire map with sequential pixel indices
region_map = np.arange(npix)

# 3. Plot the full tiling structure
hp.mollview(
    region_map, 
    title=f"Full HEALPix Tiling (NSIDE={NSIDE}, Total Tiles={npix})", 
    cmap="tab20",      # "prism" or "tab20" works great for repeating distinct tile colors
    rot=(180, 0, 0),
    cbar=False         # Hides the colorbar since the values are just tile IDs
)

hp.graticule()
plt.show()