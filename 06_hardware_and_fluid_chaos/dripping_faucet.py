"""
The Chaos Toolkit - Act 6: Hardware & Fluid Chaos
Module: Shaw Dripping Faucet Model, Capillary Pinch-off & Inter-drop Return Maps
"""
import numpy as np

def simulate_dripping_faucet(flow_rate=0.8, n_drops=600, dt=0.005, gamma=0.3, omega=2.0, x_crit=1.2):
    """
    Hybrid dynamical mass-spring dripping model (Shaw 1984):
    Mass accumulates: dm/dt = flow_rate
    Oscillator state: d2x/dt2 + gamma*dx/dt + (omega^2/m)*x = g
    Threshold: when x >= x_crit, neck pinch-off triggers:
      - Inter-drop time dt_drop recorded
      - Droplet detaches carrying fraction of mass
      - Mechanical state rebounds: x -> x0, v -> -beta * v
    """
    m = 1.0
    x = 0.0
    v = 0.0
    g = 9.81
    
    drop_times = []
    t = 0.0
    last_drop_t = 0.0
    
    while len(drop_times) < n_drops and t < 2000.0:
        # Mass accumulation
        m += flow_rate * dt
        # Acceleration
        a = g - gamma * v - (omega**2 / max(m, 0.1)) * x
        v += a * dt
        x += v * dt
        t += dt
        
        # Check pinch-off singularity
        if x >= x_crit and v > 0:
            delta_t = t - last_drop_t
            if delta_t > 0.05:  # Debounce physical threshold
                drop_times.append(delta_t)
                last_drop_t = t
                # Detachment and mass shedding
                delta_m = 0.3 * m
                m = max(0.4, m - delta_m)
                x = 0.1
                v = -0.2 * v  # Elastic meniscus rebound

    dt_series = np.array(drop_times)
    dt_n = dt_series[:-1]
    dt_next = dt_series[1:]
    return dt_series, dt_n, dt_next
