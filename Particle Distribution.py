import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

M = 1.0  # Hydrogen atom
gamma = 1.0  # friction coefficient
k_B = 1.38e-23  # Boltzmann constant
T = 300  # 26.85°C = 300K
dt = 0.01
steps = 1000
particles = 10
k = 1.0  # force constant

X = np.zeros((particles, steps, 3))
V = np.zeros((particles, steps, 3))

def potential_grad(X):
    return k * X    # U(X) = 1/2kX^2

# Langevin dynamics
for i in range(1, steps):
    R = np.random.normal(0, np.sqrt(dt), (particles, 3))
    dV = (-potential_grad(X[:, i-1]) - gamma * V[:, i-1] + np.sqrt(2 * M * gamma * k_B * T) * R) * dt / M
    V[:, i] = V[:, i-1] + dV
    X[:, i] = X[:, i-1] + V[:, i] * dt

displacements_x = X[:, 1:, 0] - X[:, :-1, 0]
displacements_y = X[:, 1:, 1] - X[:, :-1, 1]
displacements_z = X[:, 1:, 2] - X[:, :-1, 2]
all_displacements = np.concatenate((displacements_x.flatten(), displacements_y.flatten(), displacements_z.flatten()))


fig, axs = plt.subplots(3, 1, figsize=(10, 7))

for ax, data, label in zip(axs, [displacements_x.flatten(), displacements_y.flatten(), displacements_z.flatten()], ['X', 'Y', 'Z']):
    ax.hist(data, bins=50, density=True, alpha=0.6, color='g')
    mu, std = norm.fit(data)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    title = f'{label} displacements: mu = {mu:.2e}, std = {std:.2e}'
    ax.set_title(title)
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()