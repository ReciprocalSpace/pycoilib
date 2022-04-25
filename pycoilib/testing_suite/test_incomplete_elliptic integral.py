from scipy.constants import mu_0 as μ0
import numpy as np
from numpy import pi as π
from scipy import integrate
import matplotlib.pyplot as plt


Rp = 0.04
angle = π
vec_x = np.array([0., 0., 1.])
vec_y = np.array([-1., 0., 0.])

vec_0 = np.array([0.05, 0., 0.04]) - np.array([-0.05, 0., 0.])  # Pos(Line) - Pos(Arc)
vec_n = np.array([-1., 0., 0.])
ell = 0.1


def vec_p(p):  # P = Arc
    return Rp*(vec_x*np.cos(p) + vec_y*np.sin(p))


def d_vec_p(p):
    return Rp*(-vec_x*np.sin(p) + vec_y*np.cos(p))


def vec_s(s):  # S = Line
    return vec_0 + vec_n * s


def d_vec_s(s):
    return vec_n


def fct_1(s, p):
    return μ0 / (4 * π) * (d_vec_s(s) @ d_vec_p(p)) / np.linalg.norm(vec_s(s) - vec_p(p))




#####
x_u = vec_x
y_u = vec_y
θ1 = angle

s_u = vec_n
Ls = ell

s0 = np.linalg.norm(vec_0)
s0_u = vec_0/s0

def integrand(θ):
    p_u = x_u*np.cos(θ) + y_u*np.sin(θ)
    dp_u = -x_u*np.sin(θ) + y_u*np.cos(θ)

    β = (Rp**2 * (1 - (p_u@s_u)**2)
         + s0**2 * (1 - (s0_u@s_u)**2)
         - 2*Rp*s0 * (s0_u@p_u -(s_u@s0_u)*(s_u@p_u)))

    β = np.sqrt(np.clip(β, 0., np.inf))

    σ1 = Ls + s0*(s_u@s0_u) - Rp*(s_u@p_u)
    σ0 = 0. + s0*(s_u@s0_u) - Rp*(s_u@p_u)

    cst = μ0 / (4 * π) * Rp * (s_u@dp_u)

    fct = cst*(np.arcsinh(σ1/β) - np.arcsinh(σ0/β))
    return fct


result_1 = []
result_2 = []
theta = np.linspace(1e-8, angle, 10000)
for p in theta:
    output = integrate.quad(fct_1, 0, ell, (p,))
    result_1.append(output[0])
    result_2.append(integrand(p))

result_1 = np.array(result_1)
result_2 = np.array(result_2)

dx = theta[1]- theta[0]
plt.plot(theta, result_1, label=f"Double intégrale: {np.sum(result_1)*dx/1e-6:.3e} uH")
plt.plot(theta, result_2, label=f"Analytique: {np.sum(result_2)*dx/1e-6:.3e} uH")
plt.legend()
plt.axhline(0., c="k")
plt.show()