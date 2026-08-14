# -*- coding: utf-8 -*-
"""
Análisis numérico de dos capacitores en serie con resistencia
(Laboratorio 3 – Experimento I)
"""

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parámetros generales
# ---------------------------------------------------------------------
R = 1e3          # Ohm
Q1_0 = 1e-6      # C (C1 inicialmente cargado con 1 µC)
Q2_0 = 0         # C (C2 descargado)
step = 1e-4      # paso de integración
time = np.arange(0, 0.01, step)

# ---------------------------------------------------------------------
# Función de integración numérica (Euler)
# ---------------------------------------------------------------------
def integrar_circuito(C1, C2):
    Q1 = [Q1_0]
    Q2 = [Q2_0]
    for t in time[1:]:
        dQ1 = (-(Q1[-1]/R)*(1/C1 + 1/C2) + (Q1_0/(R*C2))) * step
        Q1_new = Q1[-1] + dQ1
        Q2_new = Q1_0 - Q1_new   # Conservación de la carga total
        Q1.append(Q1_new)
        Q2.append(Q2_new)
    return np.array(Q1), np.array(Q2)

# ---------------------------------------------------------------------
# CASO 1: C1 = C2 = 1 µF
# ---------------------------------------------------------------------
C1_eq = 1e-6
C2_eq = 1e-6
Q1_eq, Q2_eq = integrar_circuito(C1_eq, C2_eq)

# ---------------------------------------------------------------------
# CASO 2: C1 = 10 µF, C2 = 1 µF
# ---------------------------------------------------------------------
C1_diff = 10e-6
C2_diff = 1e-6
Q1_diff, Q2_diff = integrar_circuito(C1_diff, C2_diff)

# ---------------------------------------------------------------------
# Valores estacionarios (últimos del array)
# ---------------------------------------------------------------------
Q1_eq_final = Q1_eq[-1] * 1e6
Q2_eq_final = Q2_eq[-1] * 1e6
Q1_diff_final = Q1_diff[-1] * 1e6
Q2_diff_final = Q2_diff[-1] * 1e6

# ---------------------------------------------------------------------
# Gráfico 1: Cargas Q1 y Q2 en ambos casos
# ---------------------------------------------------------------------
plt.figure(figsize=(9,6))
plt.plot(time, Q1_eq*1e6, 'orange', label="Q1 (C1=C2=1µF)")
plt.plot(time, Q2_eq*1e6, 'blue', label="Q2 (C1=C2=1µF)")
plt.plot(time, Q1_diff*1e6, 'red', label="Q1 (C1=10µF, C2=1µF)")
plt.plot(time, Q2_diff*1e6, 'cyan', label="Q2 (C1=10µF, C2=1µF)")

# ---------------------------------------------------------------------
# Líneas horizontales: valores estacionarios de cada capacitor
# ---------------------------------------------------------------------
plt.axhline(Q1_eq_final, color='orange', linestyle='--', alpha=0.7)
plt.axhline(Q2_eq_final, color='blue', linestyle='--', alpha=0.7)
plt.axhline(Q1_diff_final, color='red', linestyle='--', alpha=0.7)
plt.axhline(Q2_diff_final, color='cyan', linestyle='--', alpha=0.7)

# Texto con los valores finales
plt.text(time[-1]*0.65, Q1_eq_final+0.02, f'{Q1_eq_final:.3f} µC', color='orange')
plt.text(time[-1]*0.65, Q2_eq_final-0.04, f'{Q2_eq_final:.3f} µC', color='blue')
plt.text(time[-1]*0.65, Q1_diff_final+0.02, f'{Q1_diff_final:.3f} µC', color='red')
plt.text(time[-1]*0.65, Q2_diff_final-0.04, f'{Q2_diff_final:.3f} µC', color='cyan')

# ---------------------------------------------------------------------
# Línea negra donde se igualan (solo visible en el caso 1)
# ---------------------------------------------------------------------
plt.axhline(0.5, color='k', linestyle="--")

# ---------------------------------------------------------------------
# Ajuste automático del eje Y
# (muestra un poco más arriba y abajo de los valores finales)
# ---------------------------------------------------------------------
all_Q = np.concatenate([Q1_eq*1e6, Q2_eq*1e6, Q1_diff*1e6, Q2_diff*1e6])
plt.ylim(all_Q.min() - 0.1, all_Q.max() + 0.1)

# ---------------------------------------------------------------------
# Etiquetas y formato
# ---------------------------------------------------------------------
plt.xlabel("Tiempo [s]", fontsize=12)
plt.ylabel("Carga [µC]", fontsize=12)
plt.title("Carga en función del tiempo para distintos capacitores", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Gráfico 2: Corriente I(t) para ambos casos
# ---------------------------------------------------------------------
I_eq = -np.diff(Q1_eq)/step
I_diff = -np.diff(Q1_diff)/step

plt.figure(figsize=(9,5))
plt.plot(time[:-1], I_eq*1e6, 'green', label="I(t) (C1=C2=1µF)")
plt.plot(time[:-1], I_diff*1e6, 'purple', label="I(t) (C1=10µF, C2=1µF)")
plt.xlabel("Tiempo [s]", fontsize=12)
plt.ylabel("Corriente [µA]", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''


"""
Experimento II - Barra móvil (versión con m=0.01 kg)
Gráficos con tiempo en microsegundos (µs)
"""

# Parámetros reales del enunciado
V0 = 10.0
L = 0.10
m = 0.01
R = 100.0
B0 = 0.1

# Coeficientes y estado estacionario
a = (B0**2 * L**2) / (m * R)
b = (B0 * L * V0) / (m * R)
v_inf = V0 / (B0 * L)

# Tiempo (en segundos)
dt = 0.1       # paso de MIL s
t_max = 450000
t = np.arange(0, t_max, dt)

v = np.zeros_like(t)
I = np.zeros_like(t)

# Integración
for n in range(len(t) - 1):
    dvdt = -a * v[n] + b
    v[n + 1] = v[n] + dvdt * dt
    I[n] = (V0 - B0 * L * v[n]) / R
I[-1] = (V0 - B0 * L * v[-1]) / R

# -------- Gráfico de velocidad --------
plt.figure(figsize=(8,5))
plt.plot(t, v, color='teal', label='v(t)')
plt.axhline(v_inf, color='gray', linestyle='--', label=f'v_inf = {v_inf:.0f} m/s')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad [m/s]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------------------------------
#CORRIENTE
#---------------------------------------

# Coeficientes
a = (B0**2 * L**2) / (m * R)
b = (B0 * L * V0) / (m * R)
v_inf = V0 / (B0 * L)

# --- Tiempo ---
dt = 0.1           # paso de 1 ms = 0.001 s
t_max = 160000      # 0.16 s (160 000 µs, igual que el gráfico)
t = np.arange(0, t_max, dt)

v = np.zeros_like(t)
I = np.zeros_like(t)

# Integración (Euler simple)
for n in range(len(t) - 1):
    dvdt = -a * v[n] + b
    v[n + 1] = v[n] + dvdt * dt
    I[n] = (V0 - B0 * L * v[n]) / R
I[-1] = (V0 - B0 * L * v[-1]) / R

# --- Gráfico de corriente ---
plt.figure(figsize=(8,5))
plt.plot(t * 1e6, I, color='darkorange', linewidth=5)  # tiempo en µs
plt.xlabel('Tiempo (µs)')
plt.ylabel('Corriente (A)')
plt.title('Corriente')
plt.grid(True)
plt.tight_layout()
plt.show()
'''
