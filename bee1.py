import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of differential equations
def bee_model(y, t):
    S, D = y
    dS_dt = np.sin(D)
    dD_dt = S - D
    return [dS_dt, dD_dt]

# Initial conditions
S0 = 0   # Initial smell intensity
D0 = 5   # Initial distance from the hive
y0 = [S0, D0]

# Time points for the simulation
t = np.linspace(0, 20, 1000)

# Solve the differential equations
solution = odeint(bee_model, y0, t)

# Extract S(t) and D(t)
S_t = solution[:, 0]
D_t = solution[:, 1]

# Plot the results
plt.figure(figsize=(14, 6))

# Distance over time
plt.subplot(1, 2, 1)
plt.plot(t, D_t, label='Distance D(t)')
plt.xlabel('Time t')
plt.ylabel('Distance from Hive D(t)')
plt.title('Bee Distance Over Time')
plt.legend()
plt.grid(True)

# Smell intensity over time
plt.subplot(1, 2, 2)
plt.plot(t, S_t, label='Smell Intensity S(t)', color='orange')
plt.xlabel('Time t')
plt.ylabel('Smell Intensity S(t)')
plt.title('Smell Intensity Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
