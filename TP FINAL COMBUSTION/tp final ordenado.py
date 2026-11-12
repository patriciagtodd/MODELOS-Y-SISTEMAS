# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:51:29 2026

@author: Aragon
"""

# -*- coding: utf-8 -*-
"""Trabajo Final - Modelos y Sistemas | Modelo reducido de combustion."""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy import signal

# %% 1. PARAMETROS, ESTADOS Y CONDICIONES INICIALES
k0, ER, q = 10**4, 5000, 100
h, Ta, alfa, beta = 0.5, 300, 0.1, 10
N = 5000

# Cada fila es [Y(0), T(0), z(0)]. La primera es la de la catedra.
X0 = np.array([[1, 300, 300],
               [0.1, 301, 301],
               [1, 350, 300],
               [0.5, 300, 300],
               [1, 300, 400]])
colores = ['blue', 'red', 'green', 'orange', 'purple']
etiquetas = [f'CI {i+1}: ({x[0]:g}, {x[1]:g}, {x[2]:g})'
             for i, x in enumerate(X0)]

# %% 2. MODELO NO LINEAL
def modelo_nolineal(t, x):
    Y, T, z = x
    reaccion = k0 * np.exp(-ER/T) * Y
    return [-reaccion,
            q*reaccion - h*(T-Ta),
            alfa*(T-z) + beta*Y]

# %% 3. EQUILIBRIO, JACOBIANO Y ESTABILIDAD
Y_s, T_s, z_s = sp.symbols('Y T z', real=True)
# Coeficientes racionales para que SymPy no introduzca redondeos.
f = sp.Matrix([-k0*sp.exp(-ER/T_s)*Y_s,
               q*k0*sp.exp(-ER/T_s)*Y_s - sp.Rational('0.5')*(T_s-Ta),
               sp.Rational('0.1')*(T_s-z_s) + beta*Y_s])
variables = [Y_s, T_s, z_s]
S = sp.solve([sp.Eq(fi, 0) for fi in f], variables, dict=True)
if not S:
    raise ValueError('No se encontraron puntos de equilibrio.')
Xcrit = np.array([[float(sol[v]) for v in variables] for sol in S])
xe = Xcrit[0]

J = f.jacobian(variables)
A_s = J.subs(dict(zip(variables, map(sp.Integer, xe))))
A = np.array(A_s, dtype=float)
autovalores = np.linalg.eigvals(A)

print('Puntos de equilibrio:', Xcrit)
print('Derivadas en el equilibrio:', modelo_nolineal(0, xe))
print('Jacobiano evaluado A:\n', A)
print('Autovalores:', autovalores)
print('Estabilidad local asintotica:', np.all(np.real(autovalores) < 0))

# %% 4. MODELO LINEALIZADO (EN DESVIACIONES)
def modelo_linealizado(t, dx):
    return A @ dx

# %% 5. SIMULACIONES Y GRAFICOS: 200 Y 1000 SEGUNDOS
# En cada intervalo se usan exactamente las mismas CI para ambos modelos.
for t_final in [200, 1000]:
    tiempo = np.linspace(0, t_final, N)
    soluciones_nl = []
    soluciones_lin = []

    for x0 in X0:
        sol_nl = solve_ivp(modelo_nolineal, [0, t_final], x0,
                           t_eval=tiempo, rtol=1e-8, atol=1e-10)
        sol_lin = solve_ivp(modelo_linealizado, [0, t_final], x0-xe,
                            t_eval=tiempo, rtol=1e-8, atol=1e-10)
        if not sol_nl.success or not sol_lin.success:
            raise RuntimeError('Fallo una integracion con solve_ivp.')
        soluciones_nl.append(sol_nl.y)
        # El linealizado trabaja con dx: recuperamos los estados originales.
        soluciones_lin.append(sol_lin.y + xe[:, None])

    # FIGURA A: los dos diagramas de fase en una misma figura.
    fig = plt.figure(figsize=(14, 6))
    for j, (nombre, soluciones) in enumerate([('No lineal', soluciones_nl),
                                               ('Linealizado', soluciones_lin)]):
        ax = fig.add_subplot(1, 2, j+1, projection='3d')
        for i, x in enumerate(soluciones):
            ax.plot(x[0], x[1], x[2], color=colores[i], lw=1.6,
                    label=etiquetas[i])
            ax.scatter(*x[:, 0], color=colores[i], marker='*', s=65)
            ax.scatter(*x[:, -1], color=colores[i], marker='o', s=20)
        ax.scatter(*xe, color='black', marker='X', s=95, label='Equilibrio')
        ax.set(xlabel='Y', ylabel='T [K]', zlabel='z [K]', title=nombre)
        ax.view_init(elev=22, azim=-60)
        if j == 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=8)
    fig.suptitle(f'Diagramas de fase 3D - 0 a {t_final} s')
    fig.subplots_adjust(left=0.01, right=0.79, wspace=0.04)
    plt.show()

    # FIGURA B: soluciones temporales de los tres estados, lado a lado.
    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for j, (nombre, soluciones) in enumerate([('No lineal', soluciones_nl),
                                               ('Linealizado', soluciones_lin)]):
        for i, x in enumerate(soluciones):
            for k in range(3):
                ax[k, j].plot(tiempo, x[k], color=colores[i],
                              lw=1.4, label=etiquetas[i])
        ax[0, j].set_title(nombre)
        ax[2, j].set_xlabel('Tiempo [s]')
    for k, estado in enumerate(['Y', 'T [K]', 'z [K]']):
        ax[k, 0].set_ylabel(estado)
        for j in range(2):
            ax[k, j].grid(True, alpha=0.3)
    fig.suptitle(f'Soluciones temporales - 0 a {t_final} s')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='lower center',
               ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    plt.show()

# %% 6. ENTRADA TERMICA Y FUNCION DE TRANSFERENCIA
# Entrada u(t) [K/s] en la ecuacion de T, salida deltaT = T - Ta.
B = np.array([[0], [1], [0]])
C = np.array([[0, 1, 0]])
D = np.array([[0]])
# Representacion en estados, como en el apunte de Modelos Fisicos.
sistema = signal.StateSpace(A, B, C, D)

s = sp.symbols('s')
H_s = sp.cancel((sp.Matrix(C) * (s*sp.eye(3)-A_s).inv()
                 * sp.Matrix(B) + sp.Matrix(D))[0])
print('Funcion de transferencia H(s):')
sp.pprint(H_s)

# Extraemos los coeficientes de la expresion simplificada (sin escribir H a mano).
num_s, den_s = sp.fraction(H_s)
num = [float(c) for c in sp.Poly(num_s, s).all_coeffs()]
den = [float(c) for c in sp.Poly(den_s, s).all_coeffs()]
H = signal.TransferFunction(num, den)

# %% 7. RESPUESTAS AL ESCALON Y AL IMPULSO
# Condiciones iniciales nulas en las variables de desviacion.
t_resp = np.linspace(0, 20, 1000)
t_escalon, y_escalon = signal.step(H, T=t_resp)
t_impulso, y_impulso = signal.impulse(H, T=t_resp)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t_escalon, y_escalon)
ax[0].set_title('Respuesta al escalon unitario')
ax[1].plot(t_impulso, y_impulso)
ax[1].set_title('Respuesta al impulso unitario')
for a in ax:
    a.set_xlabel('Tiempo [s]')
    a.set_ylabel('Variacion de temperatura')
    a.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
