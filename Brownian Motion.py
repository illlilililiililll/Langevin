import numpy as np
import matplotlib.pyplot as plt

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

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

plt.ion()
fig.show()
fig.canvas.draw()

for i in range(steps):
    ax.clear()
    for p in range(particles):
        color = 'red' if p == 0 else 'blue'
        ax.plot(X[p, :i+1, 0], X[p, :i+1, 1], X[p, :i+1, 2], color=color)
        ax.scatter(X[p, i, 0], X[p, i, 1], X[p, i, 2], color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Brownian Motion')
    fig.canvas.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()