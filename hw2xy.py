import numpy as np
import matplotlib.pyplot as plt

# Given parameters
dt = 0.01  # Time step size
T = 1  # Total time
N = int(T / dt)  # Number of time steps

# Initial conditions
x0, y0 = 0, 20  # Initial position
t = np.linspace(0, T, N+1)

# Initialize arrays to store the results
x = np.zeros(N+1)
y = np.zeros(N+1)
x[0], y[0] = x0, y0

# Euler's method to simulate the path
for i in range(N):
    x_prime = 2 * x[i] + 2 * y[i]
    y_prime = -y[i]
    x[i+1] = x[i] + dt * x_prime
    y[i+1] = y[i] + dt * y_prime

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot x(t)
plt.subplot(1, 2, 1)
plt.plot(t, x, label='x(t)', color='blue')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.title("Path of x(t) using Euler's Method")
plt.grid()
plt.legend()

# Plot y(t)
plt.subplot(1, 2, 2)
plt.plot(t, y, label='y(t)', color='orange')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.title("Path of y(t) using Euler's Method")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
