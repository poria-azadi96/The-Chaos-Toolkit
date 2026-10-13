"""
The Chaos Toolkit - Act 8: Clinical, Financial & AI Frontiers
Module: Log-Periodic Power Law Singularity (LPPLS) & Reservoir Computing Forecasting
"""
import numpy as np

def generate_lppls_bubble(tc=100.0, m=0.35, omega=6.5, A=6.0, B=-1.2, C=0.25, phi=0.0, n_pts=600):
    """
    Synthesizes financial bubble price trajectory via Sornette LPPLS model:
    ln p(t) = A + B*(tc - t)^m + C*(tc - t)^m * cos(omega * ln(tc - t) + phi)
    """
    t = np.linspace(10.0, tc - 0.5, n_pts)
    dt_c = tc - t
    log_price = A + B * (dt_c ** m) + C * (dt_c ** m) * np.cos(omega * np.log(dt_c) + phi)
    noise = np.random.normal(0.0, 0.03, n_pts)
    price = np.exp(log_price + noise)
    return t, price, log_price

class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computing) for autonomous chaotic time-series prediction:
    Dr: reservoir dimension (neurons)
    spectral_radius: largest eigenvalue of internal adjacency matrix W
    alpha: leaking rate
    """
    def __init__(self, n_inputs=3, n_reservoir=400, spectral_radius=0.95, alpha=0.8, sparsity=0.05, reg=1e-6):
        self.n_in = n_inputs
        self.Dr = n_reservoir
        self.sr = spectral_radius
        self.alpha = alpha
        self.reg = reg
        
        # Initialize internal sparse weight matrix
        w_rand = np.random.normal(0.0, 1.0, (self.Dr, self.Dr))
        mask = np.random.rand(self.Dr, self.Dr) < sparsity
        w_sparse = w_rand * mask
        # Scale to exact spectral radius
        eigvals = np.linalg.eigvals(w_sparse)
        max_eig = np.max(np.abs(eigvals))
        self.W = w_sparse * (self.sr / max(max_eig, 1e-12))
        
        # Input weights
        self.Win = np.random.uniform(-0.15, 0.15, (self.Dr, self.n_in))
        self.Wout = None

    def train(self, input_seq, target_seq, warmup=150):
        n_steps = len(input_seq)
        states = np.zeros((n_steps, self.Dr))
        r = np.zeros(self.Dr)

        for i in range(n_steps):
            u = input_seq[i]
            r = (1.0 - self.alpha) * r + self.alpha * np.tanh(self.W @ r + self.Win @ u)
            states[i] = r

        # Exclude initial transient states
        eff_states = states[warmup:]
        eff_targets = target_seq[warmup:]

        # Ridge regression readout training: Wout = Y * R^T * (R*R^T + beta*I)^-1
        ridge_eye = self.reg * np.eye(self.Dr)
        self.Wout = eff_targets.T @ eff_states @ np.linalg.inv(eff_states.T @ eff_states + ridge_eye)
        self.last_state = states[-1]

    def forecast_autonomous(self, n_forecast_steps):
        predictions = np.zeros((n_forecast_steps, self.n_in))
        r = self.last_state.copy()
        u = self.Wout @ r

        for i in range(n_forecast_steps):
            r = (1.0 - self.alpha) * r + self.alpha * np.tanh(self.W @ r + self.Win @ u)
            u = self.Wout @ r
            predictions[i] = u

        return predictions
