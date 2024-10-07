import numpy as np
import matplotlib.pyplot as plt

# Time parameters
t_max = 50
dt = 0.01
t = np.arange(0, t_max, dt)

# Differential equations
def dSdt(S, D):
    return np.sin(D)

def dDdt(S, D):
    return S - D

# Runge-Kutta 4th Order Method
def rk4_step(S, D, dt):
    k1_S = dSdt(S, D)
    k1_D = dDdt(S, D)
    
    k2_S = dSdt(S + 0.5 * dt * k1_S, D + 0.5 * dt * k1_D)
    k2_D = dDdt(S + 0.5 * dt * k1_S, D + 0.5 * dt * k1_D)
    
    k3_S = dSdt(S + 0.5 * dt * k2_S, D + 0.5 * dt * k2_D)
    k3_D = dDdt(S + 0.5 * dt * k2_S, D + 0.5 * dt * k2_D)
    
    k4_S = dSdt(S + dt * k3_S, D + dt * k3_D)
    k4_D = dDdt(S + dt * k3_S, D + dt * k3_D)
    
    S_next = S + (dt / 6.0) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
    D_next = D + (dt / 6.0) * (k1_D + 2 * k2_D + 2 * k3_D + k4_D)
    
    return S_next, D_next

# Simulation function
def simulate_bee(S0):
    S = np.zeros_like(t)
    D = np.zeros_like(t)
    S[0] = S0
    D[0] = 0.0
    
    for i in range(1, len(t)):
        S[i], D[i] = rk4_step(S[i-1], D[i-1], dt)
    
    return S, D

# Finding the minimum S0
S0_values = np.linspace(5.5, 6.5, 100)
min_S0 = None

for S0 in S0_values:
    S, D = simulate_bee(S0)
    if np.any(D >= 3 * np.pi):
        min_S0 = S0
        break

if min_S0:
    print(f"Minimum initial smell intensity S0: {min_S0:.4f}")
else:
    print("Third flower not reached within S0 range.")

# Plotting the result for minimum S0
S, D = simulate_bee(min_S0)

plt.figure(figsize=(10, 6))
plt.plot(t, D, label='Distance D(t)')
plt.axhline(y=3*np.pi, color='r', linestyle='--', label='Third Flower (D = 3π)')
plt.axhline(y=2*np.pi, color='g', linestyle='--', label='Saddle Point (D = 2π)')
plt.xlabel('Time t')
plt.ylabel('Distance D(t)')
plt.title(f'Bee Trajectory with S0 = {min_S0:.4f}')
plt.legend()
plt.grid(True)
plt.show()
