import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Parameters
R0 = 2.5     # Base rent (thousands)
a = 0.5      # Sensitivity of Rmax to population
g = 0.5      # Rent growth baseline
K = 10.0      # Carrying capacity for population
P0 = 5.0      # Initial population (millions)
R_init = 2.5  # Initial rent (thousands)
t_span = (0, 10)  # Time span in decades
t_eval_fixed = np.linspace(t_span[0], t_span[1], 300)  # Shared time grid for consistency

# Define the modified system of equations
def system(t, y):
    P, R = y
    if P <= 0:  # Avoid log(0)
        P = 1e-5
    R_max = R0 + a * np.log(P)
    dP_dt = P * (K - P) - R
    dR_dt = g * R * (1 - R / R_max)
    return [dP_dt, dR_dt]

# Solve the modified system (with rent control)
sol = solve_ivp(system, t_span, [P0, R_init], t_eval=t_eval_fixed, method='RK45')
P_sol = sol.y[0]  # Population solution
R_sol = sol.y[1]  # Rent solution
t = sol.t         # Time grid

# Define the unmodified system (no rent control)
def population_no_control(t, y):
    P, R = y
    dP_dt = P * (K - P) - R
    dR_dt = g * R  # Exponential growth
    return [dP_dt, dR_dt]

# Solve the unmodified system (no rent control)
sol_no_control = solve_ivp(population_no_control, t_span, [P0, R_init], t_eval=t_eval_fixed, method='RK45')
interp_no_control_P = interp1d(sol_no_control.t, sol_no_control.y[0], kind='linear', fill_value="extrapolate")
interp_no_control_R = interp1d(sol_no_control.t, sol_no_control.y[1], kind='linear', fill_value="extrapolate")
P_no_control_interp = interp_no_control_P(t_eval_fixed)
R_no_control_interp = interp_no_control_R(t_eval_fixed)

# ----------------------------
# Diagram 1: Time Evolution with Rent Control
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, P_sol, label="Population (millions)", linewidth=2, color='green')
plt.plot(t, R_sol, label="Rent (thousands)", linewidth=2, color='blue')
plt.title("Time Evolution of Population and Rent (With Rent Control)", fontsize=14)
plt.xlabel("Time (decades)", fontsize=12)
plt.ylabel("Population / Rent", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------
# Diagram 2: Phase Portrait with Rent Control
# ----------------------------
P_vals = np.linspace(1, K, 100)
R_vals = np.linspace(1, 10, 100)
PP, RR = np.meshgrid(P_vals, R_vals)
R_max_grid = R0 + a * np.log(PP)
dP_dt = PP * (K - PP) - RR
dR_dt = g * RR * (1 - RR / R_max_grid)

plt.figure(figsize=(10, 6))
plt.streamplot(P_vals, R_vals, dP_dt, dR_dt, density=1.5, color='blue', linewidth=1)
plt.plot(P_sol[-1], R_sol[-1], 'ro', label='Approx. Equilibrium')
plt.title("Phase Portrait: Population vs Rent Dynamics (With Rent Control)", fontsize=14)
plt.xlabel("Population (millions)", fontsize=12)
plt.ylabel("Rent (thousands)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------
# Diagram 3: Rent Growth Without Control
# ----------------------------
t_vals = np.linspace(0, 10, 300)
R_unmodified = R0 * np.exp(g * t_vals)  # Exponential rent growth

plt.figure(figsize=(10, 6))
plt.plot(t_vals, R_unmodified, label="Exponential Rent Growth (No Legislation)", linewidth=2, color='orange')
plt.title("Unmodified Rent Growth (No Rent Control)", fontsize=14)
plt.xlabel("Time (decades)", fontsize=12)
plt.ylabel("Rent (thousands)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------
# Diagram 4: Comparison of Rent Growth
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t, R_sol, label="Modified Rent Growth (With Legislation)", linewidth=2, color='blue')
plt.plot(t_vals, R_unmodified, label="Exponential Rent Growth (No Legislation)", linewidth=2, linestyle='dashed', color='orange')
plt.title("Comparison of Rent Growth: With vs Without Rent Control", fontsize=14)
plt.xlabel("Time (decades)", fontsize=12)
plt.ylabel("Rent (thousands)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------
# Diagram 5: Comparison of Population Dynamics
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval_fixed, P_no_control_interp, label="Population (No Rent Control)", linewidth=2, color='red')
plt.plot(t_eval_fixed, P_sol, label="Population (With Rent Control)", linewidth=2, color='green')
plt.title("Comparison of Population Dynamics: With vs Without Rent Control", fontsize=14)
plt.xlabel("Time (decades)", fontsize=12)
plt.ylabel("Population (millions)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# ----------------------------
# Diagram 6: Comparison of Rent Dynamics
# ----------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_eval_fixed, R_no_control_interp, label="Rent (No Rent Control)", linewidth=2, color='red')
plt.plot(t_eval_fixed, R_sol, label="Rent (With Rent Control)", linewidth=2, color='blue')
plt.title("Comparison of Rent Dynamics: With vs Without Rent Control", fontsize=14)
plt.xlabel("Time (decades)", fontsize=12)
plt.ylabel("Rent (thousands)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
