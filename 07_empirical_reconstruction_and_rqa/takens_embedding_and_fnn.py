"""
The Chaos Toolkit - Act 7: Empirical Phase Space Reconstruction & RQA
Module: Takens Delay Coordinate Embedding, AMI Lag Optimization & FNN Dimension Estimation
"""
import numpy as np

def delay_embed(time_series, tau, m):
    """
    Constructs m-dimensional delay coordinate vectors:
    y_n = [s_n, s_{n+tau}, s_{n+2*tau}, ..., s_{n+(m-1)*tau}]
    """
    n_pts = len(time_series) - (m - 1) * tau
    if n_pts <= 0:
        raise ValueError("Time series is too short for given embedding parameters.")
    embedded = np.empty((n_pts, m))
    for i in range(m):
        embedded[:, i] = time_series[i * tau : i * tau + n_pts]
    return embedded

def compute_ami(time_series, max_tau=35, n_bins=32):
    """
    Calculates Fraser-Swinney Average Mutual Information (AMI):
    I(tau) = sum P(s_n, s_{n+tau}) * log2( P(s_n, s_{n+tau}) / (P(s_n) * P(s_{n+tau})) )
    Optimal delay tau* is chosen at the first local minimum.
    """
    s_min = np.min(time_series)
    s_max = np.max(time_series) + 1e-12
    norm_s = ((time_series - s_min) / (s_max - s_min) * n_bins).astype(int)
    norm_s = np.clip(norm_s, 0, n_bins - 1)

    ami_curve = np.zeros(max_tau)

    for tau in range(1, max_tau + 1):
        s_current = norm_s[:-tau]
        s_delayed = norm_s[tau:]
        n_samples = len(s_current)

        joint_hist, _, _ = np.histogram2d(
            s_current, s_delayed, bins=n_bins, range=[[0, n_bins], [0, n_bins]]
        )
        p_joint = joint_hist / n_samples
        p_current = np.sum(p_joint, axis=1, keepdims=True)
        p_delayed = np.sum(p_joint, axis=0, keepdims=True)

        independent_prob = p_current * p_delayed
        mask = (p_joint > 0) & (independent_prob > 0)
        
        ami_val = np.sum(p_joint[mask] * np.log2(p_joint[mask] / independent_prob[mask]))
        ami_curve[tau - 1] = max(0.0, ami_val)

    # Detect first local minimum
    min_candidates = []
    for i in range(1, len(ami_curve) - 1):
        if ami_curve[i] < ami_curve[i - 1] and ami_curve[i] < ami_curve[i + 1]:
            min_candidates.append(i + 1)

    optimal_tau = min_candidates[0] if len(min_candidates) > 0 else int(np.argmin(ami_curve) + 1)
    return np.arange(1, max_tau + 1), ami_curve, optimal_tau

def compute_fnn(time_series, tau, max_m=6, r_tol=15.0, a_tol=2.0):
    """
    Computes percentage of False Nearest Neighbors (Kennel, Brown, Abarbanel 1992).
    Optimal embedding dimension m* is the first dimension where %FNN drops near 0.
    """
    sigma = np.std(time_series)
    fnn_percentages = np.zeros(max_m)

    for m in range(1, max_m + 1):
        y_m = delay_embed(time_series, tau, m)
        n_pts = len(y_m) - tau  # need room for m+1 check
        if n_pts < 100:
            break

        y_m_eval = y_m[:n_pts]
        false_neighbors = 0

        # Subsample for computational performance
        step = max(1, n_pts // 800)
        eval_indices = np.arange(0, n_pts, step)

        for i in eval_indices:
            pt = y_m_eval[i]
            dists = np.sqrt(np.sum((y_m_eval - pt)**2, axis=1))
            dists[i] = np.inf  # exclude self
            nn_idx = np.argmin(dists)
            r_d = dists[nn_idx]

            # Next coordinate difference in dimension m+1
            s_next_i = time_series[i + m * tau]
            s_next_nn = time_series[nn_idx + m * tau]
            diff_next = abs(s_next_i - s_next_nn)

            crit1 = (diff_next / (r_d + 1e-12)) > r_tol
            r_d_plus = np.sqrt(r_d**2 + diff_next**2)
            crit2 = (r_d_plus / (sigma + 1e-12)) > a_tol

            if crit1 or crit2:
                false_neighbors += 1

        fnn_percentages[m - 1] = (false_neighbors / float(len(eval_indices))) * 100.0

    optimal_m = 1
    for idx, p in enumerate(fnn_percentages):
        if p < 2.0:
            optimal_m = idx + 1
            break
        optimal_m = idx + 1

    return np.arange(1, max_m + 1), fnn_percentages, optimal_m
