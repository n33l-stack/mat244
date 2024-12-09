import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------
# Original (Unmodified) Model
# ---------------------------
# Equations:
# P'(t) = P(t)(10 - P(t)) - R(t)
# R'(t) = 0.5 R(t)

def unmodified_system(t, Y):
    P, R = Y
    dPdt = P*(10 - P) - R
    dRdt = 0.5*R
    return [dPdt, dRdt]

# Initial conditions
P0 = 5.0   # millions
R0 = 2.5   # thousand dollars
Y0 = [P0, R0]

# Solve over 0 to 10 decades
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 300)
sol_unmodified = solve_ivp(unmodified_system, t_span, Y0, t_eval=t_eval)

# Figure 1: Unmodified Model
fig1, ax1 = plt.subplots()
ax1.plot(sol_unmodified.t, sol_unmodified.y[0], label='Population (millions)')
ax1.set_xlabel('Time (decades)')
ax1.set_ylabel('Population (millions)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(sol_unmodified.t, sol_unmodified.y[1], color='red', label='Rent (k$)')
ax2.set_ylabel('Rent (thousand $)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title('Figure 1: Unmodified Model Dynamics')
ax1.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------
# Modified Model with Legislation
# ---------------------------
# New equations:
# P'(t) = P(t)(10 - P(t)) - R(t)
# R'(t) = 0.5 R(t) [1 - R(t)/(R0 + a ln(P(t)))]
# Parameters for modified model
R0_param = 2.5
a_param = 0.5

def modified_system(t, Y):
    P, R = Y
    if P <= 0:
        # Avoid log domain errors; if population ever drops below or equal to zero,
        # force a small positive value or handle it gracefully.
        P_mod = max(P, 1e-3)
    else:
        P_mod = P
    
    dPdt = P*(10 - P) - R
    R_max = R0_param + a_param * np.log(P_mod)
    # Ensure R_max > 0: If P_mod < 1, log(P_mod)<0. For simplicity assume population stays >0.
    dRdt = 0.5 * R * (1 - R / R_max)
    return [dPdt, dRdt]

# Solve modified system
sol_modified = solve_ivp(modified_system, t_span, Y0, t_eval=t_eval)

# Figure 2: Modified Model Time Evolution
fig2, ax1 = plt.subplots()
ax1.plot(sol_modified.t, sol_modified.y[0], label='Population (millions)')
ax1.set_xlabel('Time (decades)')
ax1.set_ylabel('Population (millions)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(sol_modified.t, sol_modified.y[1], color='red', label='Rent (k$)')
ax2.set_ylabel('Rent (thousand $)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_title('Figure 2: Modified Model Dynamics with Legislation')
ax1.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------
# Figure 3: Phase Portrait
# ---------------------------
# We will create a grid of (P, R) points and plot vector fields along with solution trajectories.
P_vals = np.linspace(0.5, 10, 20)
R_vals = np.linspace(0, 6, 20)
PP, RR = np.meshgrid(P_vals, R_vals)

# Compute vector field
dP = PP*(10-PP) - RR
# For vector field, ensure positivity in log:
R_max_grid = R0_param + a_param*np.log(np.maximum(PP, 1e-3))
dR = 0.5*RR*(1 - RR/R_max_grid)

fig3, ax3 = plt.subplots()
# Plot vector field arrows
ax3.quiver(PP, RR, dP, dR, color='gray', alpha=0.5)

# Plot a few solution trajectories from different initial conditions
initial_conditions = [
    [5.0, 2.5],   # our original initial condition
    [3.0, 1.0],
    [8.0, 4.0],
    [9.0, 0.5],
    [2.0, 5.0]
]

for ic in initial_conditions:
    sol = solve_ivp(modified_system, t_span, ic, max_step=0.1)
    ax3.plot(sol.y[0], sol.y[1], '-', label=f'IC: P={ic[0]}, R={ic[1]}')

ax3.set_xlabel('Population (millions)')
ax3.set_ylabel('Rent (thousand $)')
ax3.set_title('Figure 3: Phase Portrait of Modified Model')
ax3.grid(True)
ax3.legend(loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()
