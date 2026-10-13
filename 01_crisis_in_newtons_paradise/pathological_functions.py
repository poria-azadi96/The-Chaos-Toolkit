"""
The Chaos Toolkit - Act 1: Pathological Functions & Singular Counter-Examples
"""
import numpy as np

def weierstrass_function(x, a=0.5, b=13, n_terms=100):
    x_arr = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x_arr)
    for n in range(n_terms):
        coeff = a ** n
        freq = (b ** n) * np.pi
        result += coeff * np.cos(freq * x_arr)
    return result

def box_counting_dimension(time_series, n_boxes_list=None):
    if n_boxes_list is None:
        n_boxes_list = [16, 32, 64, 128, 256, 512, 1024]
    n_samples = len(time_series)
    x_coords = np.linspace(0.0, 1.0, n_samples)
    y_norm = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series) + 1e-12)
    counts = []
    for n_boxes in n_boxes_list:
        x_bins = np.clip((x_coords * (n_boxes - 1)).astype(int), 0, n_boxes - 1)
        y_bins = np.clip((y_norm * (n_boxes - 1)).astype(int), 0, n_boxes - 1)
        occupied_boxes = len(set(zip(x_bins, y_bins)))
        counts.append(occupied_boxes)
    coeffs = np.polyfit(np.log(n_boxes_list), np.log(counts), 1)
    return coeffs[0]

def generate_devils_staircase(k_val=1.0, resolution=1000):
    omega_values = np.linspace(0.0, 1.0, resolution)
    rotation_numbers = np.zeros(resolution)
    n_iterations = 1000
    n_transient = 400
    for idx, omega in enumerate(omega_values):
        theta = 0.5
        for _ in range(n_transient):
            theta = (theta + omega - (k_val / (2.0 * np.pi)) * np.sin(2.0 * np.pi * theta)) % 1.0
        lifted_theta = theta
        for _ in range(n_iterations):
            lifted_theta = lifted_theta + omega - (k_val / (2.0 * np.pi)) * np.sin(2.0 * np.pi * lifted_theta)
        rotation_numbers[idx] = (lifted_theta - theta) / n_iterations
    return omega_values, rotation_numbers
