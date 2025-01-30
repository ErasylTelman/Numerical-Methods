import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f1(theta1, theta2, w1, w2, l1, l2, m1, m2, g):
    Theta = (theta1 - theta2)%(2*np.pi)
    denom1 = l1*(1 + m1/m2 - np.cos(Theta)**2)
    return (g*(np.sin(theta2)*np.cos(Theta) - (1 + m1/m2)*np.sin(theta1)) - (w2**2*l2 + w1**2*l1*np.cos(Theta))*np.sin(Theta))/denom1

def f2(theta1, theta2, w1, w2, l1, l2, m1, m2, g):
    Theta = (theta1 - theta2)%(2*np.pi)
    denom2 = l2*(1 + m1/m2 - np.cos(Theta)**2)
    return (g*(1 + m1/m2)*(np.sin(theta1)*np.cos(Theta) - np.sin(theta2)) - ((1 + m1/m2)*w1**2*l1 + w2**2*l2*np.cos(Theta))*np.sin(Theta))/denom2

def g1(w1):
    return w1

def g2(w2):
    return w2


def rk4(func1, func2, func3, func4, n, ti, tf, l1, l2, m1, m2, g):
    h = (tf - ti)/n
    t = np.linspace(ti, tf, n+1)
    theta1 = np.zeros(n+1)
    theta2 = np.zeros(n+1)
    w1 = np.zeros(n+1)
    w2 = np.zeros(n+1)

    theta1[0] = np.pi/2
    theta2[0] = np.pi/2
    w1[0] = 0
    w2[0] = 0

    for i in range(n):
        k1 = h * func1(theta1[i], theta2[i], w1[i], w2[i], l1, l2, m1, m2, g)
        b1 = h * func2(theta1[i], theta2[i], w1[i], w2[i], l1, l2, m1, m2, g)
        p1 = h * func3(w1[i])
        q1 = h * func4(w2[i])

        k2 = h * func1(theta1[i] + p1/2, theta2[i] + q1/2, w1[i] + k1/2, w2[i] + b1/2, l1, l2, m1, m2, g)
        b2 = h * func2(theta1[i] + p1/2, theta2[i] + q1/2, w1[i] + k1/2, w2[i] + b1/2, l1, l2, m1, m2, g)
        p2 = h * func3(w1[i] + k1/2)
        q2 = h * func4(w2[i] + b1/2)

        k3 = h * func1(theta1[i] + p2/2, theta2[i] + q2/2, w1[i] + k2/2, w2[i] + b2/2, l1, l2, m1, m2, g)
        b3 = h * func2(theta1[i] + p2/2, theta2[i] + q2/2, w1[i] + k2/2, w2[i] + b2/2, l1, l2, m1, m2, g)
        p3 = h * func3(w1[i] + k2/2)
        q3 = h * func4(w2[i] + m2/2)

        k4 = h * func1(theta1[i] + p3, theta2[i] + q3, w1[i] + k3, w2[i] + b3, l1, l2, m1, m2, g)
        b4 = h * func2(theta1[i] + p3, theta2[i] + q3, w1[i] + k3, w2[i] + b3, l1, l2, m1, m2, g)
        p4 = h * func3(w1[i] + k3)
        q4 = h * func4(w2[i] + b3)

        # w1[i+1] = w1[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        # w2[i+1] = w2[i] + (b1 + 2*b2 + 2*b3 + b4)/6
        w1[i+1] = np.clip(w1[i] + (k1 + 2*k2 + 2*k3 + k4)/6, -1e10, 1e10)
        w2[i+1] = np.clip(w2[i] + (b1 + 2*b2 + 2*b3 + b4)/6, -1e10, 1e10)
        theta1[i+1] = (theta1[i] + (p1 + 2*p2 + 2*p3 + p4)/6)%(2*np.pi)
        theta2[i+1] = (theta2[i] + (q1 + 2*q2 + 2*q3 + q4)/6)%(2*np.pi)

    return t, theta1, theta2, w1, w2

l1 = 1; l2 = 1; m1 = 1; m2 = 1; g = 9.8
ti = 0; tf = 50; n = 1000

w1 = np.zeros(n+1); w2 = np.zeros(n+1)
theta1 = np.zeros(n+1); theta2 = np.zeros(n+1)
w1[0] = 0; w2[0] = 0
theta1[0] = np.pi/2; theta2[0] = np.pi/2
t, theta1, theta2, w1, w2 = rk4(f1, f2, g1, g2, n, ti, tf, l1, l2, m1, m2, g)


# plt.plot(t, theta1, label='Theta1')
# plt.plot(t, theta2, label='Theta2')
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.legend()
# plt.show()

x1, y1 = l1 * np.sin(theta1), -l1 * np.cos(theta1)
x2, y2 = x1 + l2 * np.sin(theta2), y1 - l2 * np.cos(theta2)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
line, = ax.plot([], [], 'o-', lw=2, markersize=8)

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=n, interval=30, blit=True)

plt.show()