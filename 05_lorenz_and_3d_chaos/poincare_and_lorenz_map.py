"""
The Chaos Toolkit - Act 5: Lorenz & 3D Chaos
Module: Poincare Section Extraction & 1D Lorenz Cusp Return Map
"""
import numpy as np

def extract_poincare_section(x_pts, y_pts, z_pts, z_plane=27.0, direction=1):
    """
    Extracts upward (or downward) crossings through plane z = z_plane.
    Returns x and y coordinates on the section.
    """
    sec_x = []
    sec_y = []
    for i in range(len(z_pts) - 1):
        z1, z2 = z_pts[i], z_pts[i+1]
        if direction >= 0 and z1 < z_plane and z2 >= z_plane:
            # Linear interpolation
            weight = (z_plane - z1) / (z2 - z1 + 1e-12)
            sec_x.append(x_pts[i] + weight * (x_pts[i+1] - x_pts[i]))
            sec_y.append(y_pts[i] + weight * (y_pts[i+1] - y_pts[i]))
        elif direction < 0 and z1 > z_plane and z2 <= z_plane:
            weight = (z1 - z_plane) / (z1 - z2 + 1e-12)
            sec_x.append(x_pts[i] + weight * (x_pts[i+1] - x_pts[i]))
            sec_y.append(y_pts[i] + weight * (y_pts[i+1] - y_pts[i]))
    return np.array(sec_x), np.array(sec_y)

def compute_lorenz_cusp_map(z_pts):
    """
    Extracts successive local maxima of z(t): z_n vs z_{n+1}.
    Edward Lorenz discovered this 1D cusp map to prove deterministic chaos
    in continuous differential equations without stable periodic orbits.
    """
    peaks = []
    for i in range(1, len(z_pts) - 1):
        if z_pts[i] > z_pts[i-1] and z_pts[i] > z_pts[i+1]:
            peaks.append(z_pts[i])
    
    peaks = np.array(peaks)
    z_n = peaks[:-1]
    z_next = peaks[1:]
    return z_n, z_next
