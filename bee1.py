import numpy as np
import matplotlib.pyplot as plt

# Define time parameters for the simulation
time_end = 20  # Duration of the simulation
time_steps = 500  # Number of time steps
t = np.linspace(0, time_end, time_steps)  # Time array

# Define initial conditions
S_initial = 0  # Neutral smell intensity at t=0
D_initial = 5  # Bee starts at distance 5 from its hive

# Initialize arrays to store results
S = np.zeros(time_steps)
D = np.zeros(time_steps)
S[0] = S_initial
D[0] = D_initial

# Define time step size
dt = t[1] - t[0]

# Perform the simulation using Euler's method
for i in range(1, time_steps):
    S[i] = S[i-1] + np.sin(D[i-1]) * dt
    D[i] = D[i-1] + (S[i-1] - D[i-1]) * dt

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t, S, label='Smell Intensity S(t)')
plt.plot(t, D, label='Distance from Hive D(t)', linestyle='--')
plt.xlabel('Time (t)')
plt.ylabel('Value')
plt.title('Bee Movement Simulation: Smell Intensity and Distance Over Time')
plt.legend()
plt.grid(True)
plt.show()
