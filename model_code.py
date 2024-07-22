
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Parameters
dx = 0.1  # Spatial step size
dt = 0.001  # Reduced time step size
L = 10.0  # Length of the spatial domain
T = 100  # Number of time steps

# Dropout rate
dropout_rate = 0.1  # 10% chance to drop an update

# Grid points
x = np.arange(0, L, dx)
N = len(x)

# Initial field configurations
phi = np.zeros(N)
H = np.zeros(N)
W = np.zeros(N)
Z = np.zeros(N)
A = np.zeros(N)
phi[int(N / 2)] = 1  # Initial condition for phi

# Parameters for the potential and interaction terms
g = 0.1  # Reduced coupling constant
gw = 0.05  # Reduced weak coupling constant
gz = 0.05  # Reduced Z coupling constant
ga = 0.01  # Reduced electromagnetic coupling constant

# Arrays to collect Laplacian values
laplacian_phi_arr = np.zeros((T, N))
laplacian_H_arr = np.zeros((T, N))
laplacian_W_arr = np.zeros((T, N))
laplacian_Z_arr = np.zeros((T, N))
laplacian_A_arr = np.zeros((T, N))

# Time evolution
for t in range(T):
    # Copy of the fields for updating
    phi_new = phi.copy()
    H_new = H.copy()
    W_new = W.copy()
    Z_new = Z.copy()
    A_new = A.copy()
    
    # Finite difference for the second derivative (Laplacian)
    for i in range(1, N - 1):
        laplacian_phi = (phi[i+1] - 2*phi[i] + phi[i-1]) / dx**2
        laplacian_H = (H[i+1] - 2*H[i] + H[i-1]) / dx**2
        laplacian_W = (W[i+1] - 2*W[i] + W[i-1]) / dx**2
        laplacian_Z = (Z[i+1] - 2*Z[i] + Z[i-1]) / dx**2
        laplacian_A = (A[i+1] - 2*A[i] + A[i-1]) / dx**2
        
        # Collect Laplacian values
        laplacian_phi_arr[t, i] = laplacian_phi
        laplacian_H_arr[t, i] = laplacian_H
        laplacian_W_arr[t, i] = laplacian_W
        laplacian_Z_arr[t, i] = laplacian_Z
        laplacian_A_arr[t, i] = laplacian_A
        
        # Introduce dropout
        if np.random.rand() > dropout_rate:
            # Update the fields using the discretized Euler-Lagrange equation with stability checks
            try:
                phi_new[i] = phi[i] + dt * (laplacian_phi - g * (phi[i]**2 - H[i]**2) * phi[i])
                H_new[i] = H[i] + dt * (laplacian_H - g * (H[i]**2 - phi[i]**2) * H[i])
                W_new[i] = W[i] + dt * (laplacian_W - gw * (W[i]**2 - H[i]**2) * W[i])
                Z_new[i] = Z[i] + dt * (laplacian_Z - gz * (Z[i]**2 - phi[i]**2) * Z[i])
                A_new[i] = A[i] + dt * (laplacian_A - ga * (A[i]**2 - phi[i]**2) * A[i])
            except FloatingPointError:
                continue
    
    # Update the fields
    phi = phi_new.copy()
    H = H_new.copy()
    W = W_new.copy()
    Z = Z_new.copy()
    A = A_new.copy()

# Plot the result
plt.figure(figsize=(12, 8))
plt.plot(x, phi, label='phi')
plt.plot(x, H, label='H')
plt.plot(x, W, label='W')
plt.plot(x, Z, label='Z')
plt.plot(x, A, label='A')
plt.xlabel('x')
plt.ylabel('Field value')
plt.title('Field Evolution')
plt.legend()
plt.savefig('/mnt/data/field_evolution.png')
plt.close()
