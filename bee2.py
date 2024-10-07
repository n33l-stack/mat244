import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of differential equations
def bee_system(y, t):
    S, D = y
    dSdt = np.sin(D)
    dDdt = S - D
    return [dSdt, dDdt]

# Time points
t = np.linspace(0, 50, 1000)

# Initial conditions near different flowers
initial_conditions = [
    [0, np.pi - 0.1],   # Near the first flower (n=1)
    [0, 2*np.pi - 0.1], # Near the second flower (n=2)
    [0, 3*np.pi - 0.1], # Near the third flower (n=3)
    [0, 4*np.pi - 0.1], # Near the fourth flower (n=4)
]

# Plotting phase portraits
plt.figure(figsize=(10, 6))

for idx, y0 in enumerate(initial_conditions):
    # Solve ODEs
    sol = odeint(bee_system, y0, t)
    S = sol[:, 0]
    D = sol[:, 1]
    
    # Plot trajectory in the phase plane
    plt.plot(D, S, label=f'Initial D(0) = {y0[1]:.2f}')

# Equilibrium points
n_values = np.arange(0, 5)
D_eq = n_values * np.pi
S_eq = D_eq
plt.plot(D_eq, S_eq, 'ro', label='Equilibrium Points')

plt.xlabel('Distance D(t)')
plt.ylabel('Smell Intensity S(t)')
plt.title('Phase Portraits Near Different Flowers')
plt.legend()
plt.grid(True)
plt.show()
