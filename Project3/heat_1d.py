import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def analytic_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha**2 * t)


L = 1
alpha = 1


dx = 0.01
grid_points = int(L / dx)
x = np.linspace(0, L, grid_points+1)
dt = alpha * 0.5 * dx**2
end_time = 0.4
time_points = int(end_time / dt)
print(time_points)
state = np.sin(np.pi*x)


beta = dt/dx**2
if beta > 0.5:
    print(f'Scheme will be unstable for beta value: {beta}')


plt.plot(x, state, '--k', label='initial state')


for t in range(1, time_points+1):
    state[1:grid_points] = state[1:grid_points] + beta*(state[2:]\
                            -2*state[1:grid_points] + state[0:grid_points-1])
    #state[1:grid_points] = (1 - 2*beta) * state[1:grid_points] + beta \
    #                        * (state[2:] + state[0:grid_points-1])




plt.plot(x, state, label=f'final state, euler, t={end_time}')
plt.plot(x, analytic_solution(x, end_time), '--b', label=f'analytic, t={end_time}')
plt.title(f'1D Heat Equation, dx = {dx}')
plt.legend(loc='upper right')

plt.show()
