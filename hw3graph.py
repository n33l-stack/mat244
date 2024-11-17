import numpy as np
import matplotlib.pyplot as plt

# Define the original system (O), (A_1.8), and (A_1)
def original_system(x, y):
    x_dot = -x**2 - y + 2
    y_dot = x - y
    return x_dot, y_dot

def system_A1_8(x, y):
    x_dot = -1.8 * x - y + 2.8
    y_dot = x - y
    return x_dot, y_dot

def system_A1(x, y):
    x_dot = -x - y + 2
    y_dot = x - y
    return x_dot, y_dot

# Create a meshgrid for plotting
x_vals = np.linspace(0.5, 1.5, 20)
y_vals = np.linspace(0.5, 1.5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Calculate direction fields for each system
U_original, V_original = original_system(X, Y)
U_A1_8, V_A1_8 = system_A1_8(X, Y)
U_A1, V_A1 = system_A1(X, Y)

# Set up the plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Phase Portraits Near (1, 1) for Systems (O), (A_1.8), and (A_1)")

# Plot for Original System (O)
axes[0].quiver(X, Y, U_original, V_original, color='blue')
axes[0].set_title('Original System (O)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].axvline(x=1, color='k', linestyle='--')
axes[0].axhline(y=1, color='k', linestyle='--')
axes[0].set_xlim([0.5, 1.5])
axes[0].set_ylim([0.5, 1.5])

# Plot for System (A_1.8)
axes[1].quiver(X, Y, U_A1_8, V_A1_8, color='green')
axes[1].set_title('System (A_1.8)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].axvline(x=1, color='k', linestyle='--')
axes[1].axhline(y=1, color='k', linestyle='--')
axes[1].set_xlim([0.5, 1.5])
axes[1].set_ylim([0.5, 1.5])

# Plot for System (A_1)
axes[2].quiver(X, Y, U_A1, V_A1, color='red')
axes[2].set_title('System (A_1)')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
axes[2].axvline(x=1, color='k', linestyle='--')
axes[2].axhline(y=1, color='k', linestyle='--')
axes[2].set_xlim([0.5, 1.5])
axes[2].set_ylim([0.5, 1.5])

# Show the plots
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
