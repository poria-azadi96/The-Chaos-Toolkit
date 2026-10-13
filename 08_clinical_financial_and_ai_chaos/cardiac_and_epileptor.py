"""
The Chaos Toolkit - Act 8: Clinical, Financial & AI Frontiers
Module: Cardiac HRV Poincare Geometry, Sample Entropy & Epileptor Seizure CSD
"""
import numpy as np

def generate_synthetic_rr_intervals(n_beats=1200, state="healthy"):
    """
    Synthesizes RR intervals (ms) with realistic physiological non-linear dynamics:
    - healthy: fractal 1/f-like multi-scale variability + respiratory sinus arrhythmia
    - pathological: periodic or low-variability collapse (e.g., congestive heart failure)
    """
    t = np.arange(n_beats)
    base_rr = 800.0  # ~75 bpm
    
    if state == "healthy":
        # Multi-scale sinusoidal modulations (respiratory + Mayer wave) + colored noise
        mod = 45.0 * np.sin(2.0 * np.pi * 0.25 * t / 1.0) + 25.0 * np.sin(2.0 * np.pi * 0.1 * t / 1.0)
        noise = np.random.normal(0.0, 18.0, n_beats)
        rr = base_rr + mod + noise
    else:
        # Rigid, decoupled pathological oscillations
        mod = 12.0 * np.sin(2.0 * np.pi * 0.05 * t / 1.0)
        noise = np.random.normal(0.0, 4.0, n_beats)
        rr = base_rr + mod + noise
        
    return rr

def compute_poincare_descriptors(rr_series):
    """
    Computes Poincare plot descriptors SD1 and SD2 via orthogonal coordinate rotation:
    SD1: short-term beat-to-beat variability (parasympathetic modulation)
    SD2: long-term continuous variability
    """
    rr_n = rr_series[:-1]
    rr_next = rr_series[1:]
    diff_rr = rr_next - rr_n
    sum_rr = rr_next + rr_n

    sd1 = float(np.std(diff_rr) / np.sqrt(2.0))
    sd2 = float(np.std(sum_rr) / np.sqrt(2.0))
    ratio = sd1 / (sd2 + 1e-12)

    return rr_n, rr_next, sd1, sd2, ratio

def compute_sample_entropy(time_series, m=2, r=0.2):
    """
    Calculates Sample Entropy (SampEn) with Chebyshev metric:
    SampEn = -ln( A / B )
    """
    N = len(time_series)
    r_val = r * np.std(time_series)

    def count_matches(dim):
        patterns = np.array([time_series[i : i + dim] for i in range(N - dim)])
        total_matches = 0
        n_p = len(patterns)
        for i in range(n_p):
            dists = np.max(np.abs(patterns[i+1:] - patterns[i]), axis=1)
            total_matches += np.sum(dists <= r_val)
        return total_matches

    B = count_matches(m)
    A = count_matches(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(float(A) / float(B)))

def simulate_epileptor_csd(n_steps=1800, dt=0.04):
    """
    Simulates approaching epileptic seizure transition and tracks Critical Slowing Down (CSD):
    Measures local rolling variance and lag-1 autocorrelation AR(1).
    """
    # Slow permittivity variable z drives the bifurcation
    z_drift = np.linspace(-3.5, -2.1, n_steps)
    x = np.zeros(n_steps)
    curr_x = -1.0
    
    for i in range(n_steps):
        # Cubic resting state dynamic with slow drift
        dx = (1.0 - curr_x**3 - 2.0 * curr_x**2 - z_drift[i]) + np.random.normal(0.0, 0.2)
        curr_x += dx * dt
        x[i] = curr_x

    # Compute rolling CSD indicators
    window_sz = 200
    variance_curve = np.zeros(n_steps - window_sz)
    ar1_curve = np.zeros(n_steps - window_sz)

    for i in range(len(variance_curve)):
        segment = x[i : i + window_sz]
        variance_curve[i] = np.var(segment)
        seg_n = segment[:-1] - np.mean(segment[:-1])
        seg_next = segment[1:] - np.mean(segment[1:])
        ar1 = np.sum(seg_n * seg_next) / (np.sum(seg_n**2) + 1e-12)
        ar1_curve[i] = ar1

    return x, z_drift, variance_curve, ar1_curve
