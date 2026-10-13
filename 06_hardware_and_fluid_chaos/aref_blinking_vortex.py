"""
The Chaos Toolkit - Act 6: Hardware & Fluid Chaos
Module: Aref's Blinking Vortex, Hamiltonian Particle Advection & Chaotic Mixing
"""
import numpy as np

def single_vortex_velocity(x, y, xv, yv, gamma=1.0, epsilon=0.05):
    """Calculates Hamiltonian stream velocities from a point vortex with smoothing epsilon."""
    dx = x - xv
    dy = y - yv
    r_sq = dx**2 + dy**2 + epsilon**2
    u = -(gamma / (2.0 * np.pi)) * (dy / r_sq)
    v = (gamma / (2.0 * np.pi)) * (dx / r_sq)
    return u, v

def simulate_blinking_vortex_mixing(n_particles=250, n_cycles=12, dt=0.02, T_period=1.0, vortex_dist=0.6, gamma=3.0):
    """
    Simulates tracer particles advected by Aref's blinking vortex system:
    - Half-period [0, T/2): Left vortex at (-vortex_dist, 0) active
    - Half-period [T/2, T): Right vortex at (+vortex_dist, 0) active
    """
    # Seed circular blob of dye
    theta = np.linspace(0, 2 * np.pi, n_particles)
    r = np.sqrt(np.linspace(0, 0.18**2, n_particles))
    px = 0.0 + r * np.cos(theta)
    py = 0.3 + r * np.sin(theta)

    history_x = [px.copy()]
    history_y = [py.copy()]

    steps_per_half = int((T_period / 2.0) / dt)

    for cycle in range(n_cycles):
        # Half 1: Left vortex active
        for _ in range(steps_per_half):
            u, v = single_vortex_velocity(px, py, -vortex_dist, 0.0, gamma=gamma)
            px += u * dt
            py += v * dt
        # Half 2: Right vortex active
        for _ in range(steps_per_half):
            u, v = single_vortex_velocity(px, py, vortex_dist, 0.0, gamma=gamma)
            px += u * dt
            py += v * dt

        history_x.append(px.copy())
        history_y.append(py.copy())

    return history_x, history_y
