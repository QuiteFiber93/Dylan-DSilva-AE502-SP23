import numpy as np
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from scipy import optimize
import math



#  Kepler's Equation to find E
def kepler(e: float, M: float) -> float: return lambda E: E - e*math.sin(E) - M
#  Use E to get f
def true_anomaly(e: float, M: float) -> float:
    initial_guess = M + e
    E = optimize.newton(kepler(e, M), initial_guess, maxiter=10000, tol = 1e-10)
    nu = 2 * np.arctan( np.sqrt( (1+e)/(1-e) ) * np.tan(E/2) )
    if nu < 0: nu += 2*math.pi
    elif nu > math.pi*2: nu -= 2*math.pi
    return nu

def elem2state(a, e, i, raan, aop, f, mu, radians = True):
    if not radians:
        i, raan, aop, f = np.radians(i), np.radians(raan), np.radians(aop), np.radians(f)
        
    theta = aop + f
    
    # finding r
    r_norm = a*(1-e**2)/(1+e*np.cos(f))
    r = r_norm*np.array([
        np.cos(theta)*np.cos(raan) - np.cos(i)*np.sin(raan)*np.sin(theta),
        np.cos(theta)*np.sin(raan) + np.cos(i)*np.cos(raan)*np.sin(theta),
        np.sin(i)*np.sin(theta),
    ])
    
    # finding h -> used to find v
    h = np.sqrt(mu * a * (1 - e **2))
    
    # finding v
    v = np.array([
        -mu/h * ( np.cos(raan)*(np.sin(theta) + e*np.sin(aop)) + np.sin(raan)*(np.cos(theta) + e*np.cos(aop))*np.cos(i) ),
        -mu/h * ( np.sin(raan)*(np.sin(theta) + e*np.sin(aop)) - np.cos(raan)*(np.cos(theta) + e*np.cos(aop))*np.cos(i) ),
        mu/h * ( (np.cos(theta) + e*np.cos(aop))*np.sin(i) )
    ])
    
    return r, v

def l_dot(l, g, h, L, G, H, omega = 0):
    return 1/L**3

def g_dot(l, g, h, L, G, H, omega = 0):
    return 0

def h_dot(l, g, h, L, G, H, omega = 0):
    return omega

def L_dot(l, g, h, L, G, H, omega = 0):
    return 0

def G_dot(l, g, h, L, G, H, omega = 0):
    return 0

def H_dot(l, g, h, L, G, H, omega = 0):
    return 0


def l_t(t, l, g, h, L, G, H, omega = 0, l_0 = 0):
    return 1/L**3 * t + l_0

def g_t(t, l, g, h, L, G, H, omega = 0, g_0 = 0):
    return g_0 + t*0

def h_t(t, l, g, h, L, G, H, omega = 0, h_0 = 0):
    return omega*t + h_0

w = 0.01

a = 1.
e = 0.0
i = np.radians(0)
omega = 0
Omega = 0
M = 0

l_0 = M
g_0 = omega
h_0 = Omega

L = np.sqrt(a**3)
G = L * np.sqrt(1 - e**2)
H = G * np.cos(i)

print(f"\tInitial Conditions\n------------------------------------")
print(f"a  = {a:0.5g} DU \t\t e = {e:0.5g}")
print(f"i  = {i:0.5g} rad \t M = {e:0.5g} rad")
print(f"w  = {omega:0.5g} rad \t\t Omega = {Omega:0.5g} rad")
print(f"\n\tDelaunay Variables\n------------------------------------")
print(f"l  = {l_0:0.5g} \t\t L = {L:0.5g}")
print(f"g  = {g_0:0.5g} \t\t G = {G:0.5g}")
print(f"h  = {h_0:0.5g} \t\t H = {H:0.5g}")

time = np.linspace(start = 0., stop = 100., num = 301)
l = l_0
g = g_0
h = h_0

l_vals = l_t(time, l, g, h, L, G, H, omega = w, l_0 = l_0)
g_vals = g_t(time, l, g, h, L, G, H, omega = w, g_0 = g_0)
h_vals = h_t(time, l, g, h, L, G, H, omega = w, h_0 = g_0)
L_vals = [L for t in time]
G_vals = [G for t in time]
H_vals = [H for t in time]

r_t = []

for j,t in enumerate(time):
    a = L_vals[j]**(2/3)
    e = np.sqrt(1 - (G_vals[j]/L_vals[j])**2)
    i = np.arccos(H_vals[j]/G_vals[j])
    
    M = l_vals[j]
    omega = g_vals[j]
    Omega = h_vals[j]
    
    f = true_anomaly(e, M)

    r = elem2state(a, e, i, Omega, omega, f, mu = 1)[0]
    r_t.append(r)

fig = plt.figure()
ax = plt.axes(projection='3d')

x_t = [r_t[i][0] for i in range(len(r_t))]
y_t = [r_t[i][1] for i in range(len(r_t))]
z_t = [r_t[i][2] for i in range(len(r_t))]


p = ax.plot(x_t, y_t, z_t)
ax.set_title(r"$\omega = $" + f"{w}")
ax.set_xlabel("x [DU]")
ax.set_ylabel("y [DU]")
ax.set_zlabel("z [DU]")
plt.show()