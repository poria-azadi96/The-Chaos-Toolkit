"""
The Chaos Toolkit - Act 7: Empirical Phase Space Reconstruction & RQA
Module: Recurrence Plot Generation & Recurrence Quantification Analysis (RQA)
"""
import numpy as np
from scipy.spatial.distance import cdist

def compute_recurrence_matrix(embedded_pts, eps_radius=None, fixed_rr=None):
    """
    Computes unthresholded distance matrix and binary recurrence matrix:
    R_{i,j}(eps) = Theta(eps - ||x_i - x_j||)
    """
    dist_matrix = cdist(embedded_pts, embedded_pts, metric="euclidean")
    
    if fixed_rr is not None:
        eps_radius = np.percentile(dist_matrix, fixed_rr * 100.0)
    elif eps_radius is None:
        eps_radius = 0.1 * np.max(dist_matrix)

    rec_matrix = (dist_matrix <= eps_radius).astype(int)
    return dist_matrix, rec_matrix, eps_radius

def extract_rqa_metrics(rec_matrix, l_min=2, v_min=2):
    """
    Computes canonical RQA metrics:
    - Recurrence Rate (RR)
    - Determinism (DET)
    - Max Diagonal Length (L_max) -> Divergence ~ 1 / L_max
    - Shannon Entropy of Diagonal Lines (ENTR)
    - Laminarity (LAM)
    - Trapping Time (TT)
    """
    N = rec_matrix.shape[0]
    total_points = np.sum(rec_matrix) - N  # exclude line of identity
    rr = float(total_points) / max(1, (N * (N - 1)))

    # Diagonal Line Analysis
    diag_lengths = []
    for offset in range(1, N):
        diag = np.diagonal(rec_matrix, offset=offset)
        curr_len = 0
        for val in diag:
            if val == 1:
                curr_len += 1
            elif curr_len > 0:
                diag_lengths.append(curr_len)
                curr_len = 0
        if curr_len > 0:
            diag_lengths.append(curr_len)

    diag_lengths = np.array(diag_lengths) if len(diag_lengths) > 0 else np.array([0])
    valid_diags = diag_lengths[diag_lengths >= l_min]
    
    det = float(np.sum(valid_diags) * 2) / max(1e-12, total_points) if total_points > 0 else 0.0
    l_max = int(np.max(diag_lengths)) if len(diag_lengths) > 0 else 0
    divergence = 1.0 / l_max if l_max > 0 else 0.0

    # Diagonal length Shannon entropy
    if len(valid_diags) > 0:
        counts = np.bincount(valid_diags)
        probs = counts[counts > 0] / float(len(valid_diags))
        entr = -np.sum(probs * np.log(probs))
    else:
        entr = 0.0

    # Vertical Line Analysis (Laminarity & Trapping Time)
    vert_lengths = []
    for col in range(N):
        curr_len = 0
        for row in range(N):
            if row != col and rec_matrix[row, col] == 1:
                curr_len += 1
            elif curr_len > 0:
                vert_lengths.append(curr_len)
                curr_len = 0
        if curr_len > 0:
            vert_lengths.append(curr_len)

    vert_lengths = np.array(vert_lengths) if len(vert_lengths) > 0 else np.array([0])
    valid_verts = vert_lengths[vert_lengths >= v_min]
    
    lam = float(np.sum(valid_verts)) / max(1e-12, total_points) if total_points > 0 else 0.0
    tt = float(np.mean(valid_verts)) if len(valid_verts) > 0 else 0.0

    return {
        "RR": rr,
        "DET": min(1.0, det),
        "L_max": l_max,
        "DIV": divergence,
        "ENTR": entr,
        "LAM": min(1.0, lam),
        "TT": tt
    }
