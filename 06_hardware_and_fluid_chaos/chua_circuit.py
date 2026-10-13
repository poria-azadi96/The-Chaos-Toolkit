"""
The Chaos Toolkit - Act 6: Hardware & Fluid Chaos
Module: Chua's Circuit Dynamics, Shilnikov Chaos & Pecora-Carroll Synchronization
"""
import numpy as np
from scipy.integrate import solve_ivp

def chua_diode_characteristic(x, m0=-1.143, m1=-0.714):
    """Piecewise-linear Chua's diode characteristic: f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)"""
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))

def chua_master_derivatives(t, state, alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714):
    """
    Standard dimensionless Chua system:
    dx/dt = alpha * (y - x - f(x))
    dy/dt = x - y + z
    dz/dt = -beta * y
    """
    x, y, z = state
    fx = chua_diode_characteristic(x, m0, m1)
    dx = alpha * (y - x - fx)
    dy = x - y + z
    dz = -beta * y
    return [dx, dy, dz]

def simulate_chua_master(state0=(0.7, 0.0, 0.0), t_span=(0.0, 50.0), dt=0.01, alpha=15.6, beta=28.0):
    """Simulates Master Chua circuit with high precision."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        chua_master_derivatives, t_span, state0,
        args=(alpha, beta), t_eval=t_eval, rtol=1e-8, atol=1e-8
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

def simulate_pecora_carroll_masking(t_span=(0.0, 40.0), dt=0.01, alpha=15.6, beta=28.0, msg_amp=0.08, msg_freq=2.0):
    """
    Implements Master-Slave Pecora-Carroll Chaotic Masking:
    - Master sends: s(t) = x_m(t) + message(t)
    - Slave receiver (y_s, z_s) is driven by s(t):
        dy_s/dt = s(t) - y_s + z_s
        dz_s/dt = -beta * y_s
    - Recovered message: msg_rec(t) = s(t) - x_s_est(t) ~ s(t) - (s(t) - (dy_s/dt - z_s))
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    # 1. Simulate Master
    sol_m = solve_ivp(
        chua_master_derivatives, t_span, [0.7, 0.0, 0.0],
        args=(alpha, beta), t_eval=t_eval, rtol=1e-8, atol=1e-8
    )
    xm, ym, zm = sol_m.y[0], sol_m.y[1], sol_m.y[2]
    
    # 2. Informational message
    message = msg_amp * np.sin(2.0 * np.pi * msg_freq * t_eval)
    transmitted = xm + message

    # 3. Stepwise simulate Slave receiver subsystem
    ys = np.zeros_like(t_eval)
    zs = np.zeros_like(t_eval)
    ys[0], zs[0] = 0.5, 0.2  # arbitrary slave initial state
    
    for i in range(len(t_eval) - 1):
        s_val = transmitted[i]
        dys = s_val - ys[i] + zs[i]
        dzs = -beta * ys[i]
        ys[i+1] = ys[i] + dys * dt
        zs[i+1] = zs[i] + dzs * dt

    # Recovered state estimate
    recovered_msg = ym - ys  # Synchronized state difference reveals hidden message

    return t_eval, xm, transmitted, message, recovered_msg
