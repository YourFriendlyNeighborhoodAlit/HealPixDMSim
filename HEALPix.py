"""
4-region HEALPix tiling displayed in horizontal (alt-az) coordinates.
Uses a direct az/alt raster instead of mollview to guarantee alignment.
"""

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
import astropy.units as u
import os

# ── Configuration ─────────────────────────────────────────────────────────────

NSIDE    = 64
OBS_LAT  =  51.5
OBS_LON  =  -0.1
OBS_ELEV =  10.0
OBS_TIME = "2024-06-21 01:00:00"

POLE_RA  = 310.358
POLE_DEC =  45.280

ECL_RA   = 270.0
ECL_DEC  =  66.56

BELT_HALF_WIDTH = np.degrees(np.arcsin(0.5))  # 30°


# ── Setup ─────────────────────────────────────────────────────────────────────

t     = Time(OBS_TIME)
loc   = EarthLocation(lon=OBS_LON*u.deg, lat=OBS_LAT*u.deg, height=OBS_ELEV*u.m)
frame = AltAz(obstime=t, location=loc)

def rotation_matrix_z_to(v):
    v = v / np.linalg.norm(v)
    z = np.array([0.,0.,1.])
    ax = np.cross(z, v); n = np.linalg.norm(ax)
    if n < 1e-9: return np.eye(3) if v[2]>0 else np.diag([1.,-1.,-1.])
    ax /= n; a = np.arccos(np.clip(np.dot(z,v),-1,1))
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K@K)

def azel_to_vec(az_deg, alt_deg):
    az = np.radians(az_deg); alt = np.radians(alt_deg)
    return np.array([np.cos(alt)*np.cos(az), np.cos(alt)*np.sin(az), np.sin(alt)])

def radec_to_azel(ra_deg, dec_deg):
    sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame=ICRS())
    aa = sc.transform_to(frame)
    return aa.az.deg, aa.alt.deg


# ── Poles in horizontal ───────────────────────────────────────────────────────

pole_az, pole_alt = radec_to_azel(POLE_RA, POLE_DEC)
ecl_az,  ecl_alt  = radec_to_azel(ECL_RA,  ECL_DEC)

R_pole = rotation_matrix_z_to(azel_to_vec(pole_az, pole_alt))
R_ecl  = rotation_matrix_z_to(azel_to_vec(ecl_az,  ecl_alt))

print(f"Cygnus pole:   az={pole_az:.1f}°  alt={pole_alt:.1f}°")
print(f"Ecliptic pole: az={ecl_az:.1f}°  alt={ecl_alt:.1f}°")


# ── Build HEALPix region map (equatorial indexed) ─────────────────────────────

NPIX    = hp.nside2npix(NSIDE)
eq_vecs = np.stack(hp.pix2vec(NSIDE, np.arange(NPIX)), axis=1)
ra_all  = np.degrees(np.arctan2(eq_vecs[:,1], eq_vecs[:,0])) % 360
dec_all = np.degrees(np.arcsin(np.clip(eq_vecs[:,2], -1, 1)))

coords  = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, frame=ICRS())
aa_all  = coords.transform_to(frame)

hz_vecs = np.column_stack([
    np.cos(np.radians(aa_all.alt.deg)) * np.cos(np.radians(aa_all.az.deg)),
    np.cos(np.radians(aa_all.alt.deg)) * np.sin(np.radians(aa_all.az.deg)),
    np.sin(np.radians(aa_all.alt.deg))
])

pole_lat = np.degrees(np.arcsin(np.clip((R_pole.T @ hz_vecs.T)[2], -1, 1)))
ecl_lat  = np.degrees(np.arcsin(np.clip((R_ecl.T  @ hz_vecs.T)[2], -1, 1)))
in_belt  = np.abs(ecl_lat) < BELT_HALF_WIDTH

region_map = np.where(
    in_belt & (pole_lat >= 0), 1,
    np.where(in_belt & (pole_lat < 0), 2,
    np.where(pole_lat >= 0, 0, 3))
).astype(float)

print()
for i, lbl in enumerate(["Cygnus cap","Belt (Cygnus)","Belt (anti-Cygnus)","Anti-Cygnus cap"]):
    n = int(np.sum(region_map == i))
    print(f"  Region {i}  {lbl:22s}  {n} px  ({100*n/NPIX:.2f}%)")


# ── Rasterise onto az/alt grid (Mollweide projection) ────────────────────────
# Build a dense az/alt grid, look up each point's HEALPix pixel directly
# in horizontal coordinates — no coordinate transformation needed.

def classify_azel(az_deg, alt_deg):
    """Classify (az, alt) arrays directly using the rotation matrices."""
    v = np.column_stack([
        np.cos(np.radians(alt_deg)) * np.cos(np.radians(az_deg)),
        np.cos(np.radians(alt_deg)) * np.sin(np.radians(az_deg)),
        np.sin(np.radians(alt_deg))
    ])
    pl = (R_pole.T @ v.T)[2]
    el = (R_ecl.T  @ v.T)[2]
    ib = np.abs(el) < np.sin(np.radians(BELT_HALF_WIDTH))
    return np.where(ib & (pl>=0), 1,
           np.where(ib & (pl< 0), 2,
           np.where(pl>=0, 0, 3)))

# Mollweide grid in az/alt
W, H = 1200, 600
# Mollweide: x in [-2,2], y in [-1,1]
xs = np.linspace(-2, 2, W)
ys = np.linspace(-1, 1, H)
Xg, Yg = np.meshgrid(xs, ys)

# Mollweide inverse projection
theta_m = np.arcsin(np.clip(Yg, -1, 1))
lam = np.pi * Xg / (2 * np.cos(theta_m))   # longitude in [-pi, pi]
lat = np.arcsin(np.clip((2*theta_m + np.sin(2*theta_m)) / np.pi, -1, 1))

# Mask outside ellipse
inside = (Xg/2)**2 + Yg**2 <= 1

# Map longitude to azimuth: center at south (az=180°), going right = decreasing az
# lon=0 → az=180, lon=pi → az=0 (or 360), lon=-pi → az=360
az_grid  = (180 - np.degrees(lam)) % 360
alt_grid = np.degrees(lat)

# Classify
az_f   = az_grid[inside]
alt_f  = alt_grid[inside]
reg_f  = classify_azel(az_f, alt_f)

img = np.full((H, W), np.nan)
img[inside] = reg_f


# ── Plot ──────────────────────────────────────────────────────────────────────

COLORS = ['#5B9BD5', '#7BC8A4', '#C8E6A0', '#E8995A']
cmap   = mcolors.ListedColormap(['white'] + COLORS)  # 0=white (outside)
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm   = mcolors.BoundaryNorm(bounds, len(COLORS)+1)

fig, ax = plt.subplots(figsize=(13, 7))
ax.imshow(img, extent=[-2, 2, -1, 1], origin="lower",
          cmap=mcolors.ListedColormap(COLORS),
          norm=mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],4),
          aspect="equal", interpolation="nearest")

# Ellipse border
theta_e = np.linspace(0, 2*np.pi, 500)
ax.plot(2*np.cos(theta_e), np.sin(theta_e), 'k-', linewidth=1.5)

# Graticule in Mollweide
def azel_to_moll(az_deg, alt_deg):
    """Convert az/alt to Mollweide xy with south at center."""
    lam_r = np.radians((180 - az_deg + 360) % 360)
    lam_r = np.where(lam_r > np.pi, lam_r - 2*np.pi, lam_r)
    phi_r = np.radians(alt_deg)
    # Newton solve for Mollweide theta, clamp near poles to avoid singularity
    th = np.array(phi_r, dtype=float).copy()
    th = np.clip(th, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    for _ in range(50):
        denom = 2 + 2*np.cos(2*th)
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        th -= (2*th + np.sin(2*th) - np.pi*np.sin(phi_r)) / denom
        th = np.clip(th, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
    x = 2*lam_r*np.cos(th)/np.pi
    y = np.sin(th)
    return x, y

# Altitude lines
for alt in range(-60, 90, 30):
    az_r = np.linspace(0, 360, 500)
    x, y = azel_to_moll(az_r, np.full(500, alt))
    ok = (x/2)**2 + y**2 <= 1.001
    ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan),
            'k:', linewidth=0.4, alpha=0.4)

# Azimuth lines
for az in range(0, 360, 30):
    alt_r = np.linspace(-90, 90, 300)
    x, y  = azel_to_moll(np.full(300, az), alt_r)
    ax.plot(x, y, 'k:', linewidth=0.4, alpha=0.4)
ax.axhline(0, color='k', linewidth=0.7, alpha=0.4)   # horizon
ax.axvline(0, color='k', linewidth=0.7, alpha=0.4)   # meridian

# Axis labels
for az, lbl in [(0,'N'),(90,'E'),(180,'S'),(270,'W')]:
    x, y = azel_to_moll(az, 0)
    ax.text(x, y-0.07, lbl, ha='center', va='top', fontsize=9, alpha=0.6)
for alt in [-60,-30,0,30,60]:
    x, y = azel_to_moll(180, alt)
    ax.text(x+0.05, y, f'{alt}°', ha='left', va='center', fontsize=7, alpha=0.5)

# Belt boundaries
belt_lons = np.linspace(0, 2*np.pi, 2000)
for sign in [1, -1]:
    lat_rad = np.radians(sign * BELT_HALF_WIDTH)
    pts_ecl = np.column_stack([
        np.cos(lat_rad)*np.cos(belt_lons),
        np.cos(lat_rad)*np.sin(belt_lons),
        np.full(len(belt_lons), np.sin(lat_rad))
    ])
    pts_hz = (R_ecl @ pts_ecl.T).T
    b_alt  = np.degrees(np.arcsin(np.clip(pts_hz[:,2], -1, 1)))
    b_az   = np.degrees(np.arctan2(pts_hz[:,1], pts_hz[:,0])) % 360
    bx, by = azel_to_moll(b_az, b_alt)
    # Break line at wrap-arounds
    jump = np.abs(np.diff(bx)) > 0.3
    bx[:-1][jump] = np.nan
    ax.plot(bx, by, 'k--', linewidth=1.3, alpha=0.8)

# Cygnus markers
for az, alt, col, lbl in [
    (pole_az, pole_alt, '#1a5fa8', f'Cygnus  az={pole_az:.0f}°  alt={pole_alt:.0f}°'),
    ((pole_az+180)%360, -pole_alt, '#b85c00', f'Anti-Cygnus  az={(pole_az+180)%360:.0f}°  alt={-pole_alt:.0f}°')
]:
    x, y = azel_to_moll(az, alt)
    ax.scatter(x, y, marker='*', s=400, c=col, zorder=10, label=lbl)

# Colorbar
cax = fig.add_axes([0.15, 0.04, 0.7, 0.025])
cb  = plt.colorbar(plt.cm.ScalarMappable(
        cmap=mcolors.ListedColormap(COLORS),
        norm=mcolors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],4)),
        cax=cax, orientation='horizontal', ticks=[0,1,2,3])
cb.ax.set_xticklabels(
    ['Cygnus cap','Belt (Cygnus)','Belt (anti-Cygnus)','Anti-Cygnus cap'], fontsize=9)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02),
          ncol=2, fontsize=9, frameon=True)
ax.set_title(
    f"4-region tiling — horizontal Mollweide  |  "
    f"lat={OBS_LAT}°N  lon={OBS_LON}°E  |  {OBS_TIME} UTC", fontsize=11)
ax.axis('off')

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healpix_final.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved → {out}")