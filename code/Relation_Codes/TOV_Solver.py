import numpy as np
from numba import jit

# Constants
G = 1
c = 1
M01 = 1.474 * 10**3

# Gravitational constant and speed of light in SI units
G1 = 6.67 * 10**(-11)  # N m^2/kg^2
c1 = 3 * 10**8  # m/s

# Conversion factor from MeV/fm^3 to natural units
Mev_fm3_to_GU = 1.6 * 10**32 * G1 / c1**4

# Helper functions
@jit(nopython=True)
def find_index(arr, val):
    for i, item in enumerate(arr):
        if val > item:
            continue
        else:
            return i
    return len(arr)

@jit(nopython=True)
def linear_interp(x_arr, y_arr, x):
    if x < min(x_arr) or x > max(x_arr):
        return 0
    ind = find_index(x_arr, x)
    x1, x2 = x_arr[ind - 1], x_arr[ind]
    y1, y2 = y_arr[ind - 1], y_arr[ind]
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

@jit(nopython=True)
def mass_eqn(r, density):
    return 4 * np.pi * r**2 * density

@jit(nopython=True)
def tov_eqn(P, r, m, density, pressure, min_pressure):
    if P < min_pressure:
        return 0.0
    eden = linear_interp(pressure, density, P)
    return -(G * ((P / c**2) + eden) * (m + 4 * np.pi * r**3 * P / c**2)) / (r * (r - 2 * G * m / c**2))

def love_number(C, y):
    epsilon = 1e-6  # Small number to avoid zero division
    denom = (1 - 2 * C) + epsilon  # Add epsilon to prevent singularity
    k2 = 8 / 5 * C**5 * denom**2 * (2 + 2 * C * (y - 1) - y) / (
        2 * C * (6 - 3 * y + 3 * C * (5 * y - 8)) +
        4 * C**3 * (13 - 11 * y + C * (3 * y - 2) + 2 * C**2 * (1 + y)) +
        3 * denom**2 * (2 - y + 2 * C * (y - 1)) * np.log(denom))
    return k2

def tov_solver(cen_dens, pressure, density, exit_pressure, love=False):
    radii, masses, compacts, love_numbers = [], [], [], []

    for central_density in cen_dens:
        P = linear_interp(density, pressure, central_density)
        r, m = 10, 0
        h = 1  # Step size
        
        # Variables for Love numbers
        if love:
            H = r**2
            beta = 2 * r

        while P > exit_pressure:
            k1_m = mass_eqn(r, linear_interp(pressure, density, P))
            k1_p = tov_eqn(P, r, m, density, pressure, exit_pressure)

            k2_m = mass_eqn(r + h / 2, linear_interp(pressure, density, P))
            k2_p = tov_eqn(P + k1_p * h / 2, r + h / 2, m + k1_m * h / 2, density, pressure, exit_pressure)

            k3_m = mass_eqn(r + h / 2, linear_interp(pressure, density, P))
            k3_p = tov_eqn(P + k2_p * h / 2, r + h / 2, m + k2_m * h / 2, density, pressure, exit_pressure)

            k4_m = mass_eqn(r + h, linear_interp(pressure, density, P))
            k4_p = tov_eqn(P + k3_p * h, r + h, m + k3_m * h, density, pressure, exit_pressure)

            P += h * (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6
            m += h * (k1_m + 2 * k2_m + 2 * k3_m + k4_m) / 6
            r += h

            if love:
                k1_beta, k1_H = beta, H
                beta += k1_beta
                H += k1_H

        radii.append(r / 1000)  # Convert to km
        masses.append(m / M01)  # Convert to solar masses

        if love:
            C = m / r
            y = r * beta / H
            compacts.append(C)
            love_numbers.append(love_number(C, y))

    return radii, masses, compacts, love_numbers

# Example usage
def generate_mr_curve(E_arr, P_arr):
    p_arr = Mev_fm3_to_GU * P_arr
    e_arr = Mev_fm3_to_GU * E_arr
    
    cen_densities = np.logspace(np.log10(min(e_arr) * 1.1), np.log10(max(e_arr) * 0.99), 80)
    return tov_solver(cen_densities, p_arr, e_arr, min(p_arr) * 1.1, love=False)

# Generate M-R curve
#radii, masses, compacts, love_nums = generate_mr_curve(E_arr, P_arr)