# -*- coding: utf-8 -*-
"""
Gráficos con marcadores de estado estacionario
Experimento I (dos capacitores en serie) y Experimento II (barra móvil)
"""
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Experimento I: capacitores
# ----------------------------
R = 1e3         # ohm
Q1_0 = 1e-6     # C (1 µC)
step = 1e-4
time = np.arange(0, 0.2, step)

def integrar_circuito(C1, C2):
    Q1 = [Q1_0]
    Q2 = [0.0]
    for _ in time[1:]:
        dQ1 = (-(Q1[-1]/R)*(1/C1 + 1/C2) + (Q1_0/(R*C2))) * step
        Q1_new = Q1[-1] + dQ1
        Q2_new = Q1_0 - Q1_new
        Q1.append(Q1_new)
        Q2.append(Q2_new)
    return np.array(Q1), np.array(Q2)

# Caso 1: C1 = C2 = 1 µF
C1_eq = 1e-6
C2_eq = 1e-6
Q1_eq, Q2_eq = integrar_circuito(C1_eq, C2_eq)
# Valores estacionarios analíticos
Q1_eq_inf = Q1_0 * C1_eq / (C1_eq + C2_eq)
Q2_eq_inf = Q1_0 - Q1_eq_inf

# Caso 2: C1 = 10 µF, C2 = 1 µF
C1_diff = 10e-6
C2_diff = 1e-6
Q1_diff, Q2_diff = integrar_circuito(C1_diff, C2_diff)
Q1_diff_inf = Q1_0 * C1_diff / (C1_diff + C2_diff)
Q2_diff_inf = Q1_0 - Q1_diff_inf

# Plot cargas con marcadores de estado estacionario
plt.figure(figsize=(9,6))
plt.plot(time, Q1_eq*1e6, color='orange', label="Q1 (C1=C2=1µF)")
plt.plot(time, Q2_eq*1e6, color='blue', label="Q2 (C1=C2=1µF)")
plt.plot(time, Q1_diff*1e6, color='red', linestyle='--', label="Q1 (C1=10µF, C2=1µF)")
plt.plot(time, Q2_diff*1e6, color='cyan', linestyle='--', label="Q2 (C1=10µF, C2=1µF)")

# marcar estados estacionarios (convertidos a µC)
plt.scatter(time[-1], Q1_eq_inf*1e6, color='orange', edgecolor='k', zorder=5)
plt.scatter(time[-1], Q2_eq_inf*1e6, color='blue', edgecolor='k', zorder=5)
plt.scatter(time[-1], Q1_diff_inf*1e6, color='red', edgecolor='k', zorder=5)
plt.scatter(time[-1], Q2_diff_inf*1e6, color='cyan', edgecolor='k', zorder=5)

# Anotaciones con valores numéricos
plt.annotate(f"Q1_inf = {Q1_eq_inf*1e6:.3f} µC", xy=(time[-1], Q1_eq_inf*1e6),
             xytext=(time[-1]*0.6, Q1_eq_inf*1e6 + 0.05),
             arrowprops=dict(arrowstyle="->"), fontsize=10)
plt.annotate(f"Q2_inf = {Q2_eq_inf*1e6:.3f} µC", xy=(time[-1], Q2_eq_inf*1e6),
             xytext=(time[-1]*0.6, Q2_eq_inf*1e6 - 0.15),
             arrowprops=dict(arrowstyle="->"), fontsize=10)

plt.annotate(f"Q1_inf = {Q1_diff_inf*1e6:.3f} µC", xy=(time[-1], Q1_diff_inf*1e6),
             xytext=(time[-1]*0.85, Q1_diff_inf*1e6 + 0.15),
             arrowprops=dict(arrowstyle="->"), fontsize=10)
plt.annotate(f"Q2_inf = {Q2_diff_inf*1e6:.3f} µC", xy=(time[-1], Q2_diff_inf*1e6),
             xytext=(time[-1]*0.85, Q2_diff_inf*1e6 - 0.15),
             arrowprops=dict(arrowstyle="->"), fontsize=10)

plt.title("Evolución temporal de las cargas Q1 y Q2 (marcas de estado estacionario)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Carga [µC]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Gráfico I(t) para capacitores: marcar I_inf = 0
# ----------------------------
I_eq = -np.diff(Q1_eq)/step
I_diff = -np.diff(Q1_diff)/step

plt.figure(figsize=(9,5))
plt.plot(time[:-1], I_eq*1e6, color='green', label="I(t) (C1=C2=1µF)")
plt.plot(time[:-1], I_diff*1e6, color='purple', linestyle='--', label="I(t) (C1=10µF, C2=1µF)")

# marcar I(t->inf) = 0
plt.hlines(0, time[0], time[-1], colors='k', linestyles=':', label="I_inf = 0 A")
plt.annotate("I_inf = 0", xy=(time[-1]*0.6, 0.0), xytext=(time[-1]*0.4, 0.05),
             arrowprops=dict(arrowstyle="->"), fontsize=10)

plt.title("Corriente en función del tiempo (capacitores)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Corriente [µA]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Experimento II: barra móvil
# ----------------------------
# Parámetros SI
V0 = 10.0
L = 0.10
m = 0.01
R_bar = 100.0
B0 = 0.1

# coeficientes y estado estacionario analítico
a = (B0**2 * L**2) / (m * R_bar)
b = (B0 * L * V0) / (m * R_bar)
v_inf = V0 / (B0 * L)   # velocidad terminal
# integración numérica Euler
dt = 1e-5
t_max = 0.5
time2 = np.arange(0, t_max, dt)
v = np.zeros_like(time2)
I = np.zeros_like(time2)
for n in range(len(time2)-1):
    dvdt = -a * v[n] + b
    v[n+1] = v[n] + dvdt * dt
    I[n] = (V0 - B0 * L * v[n]) / R_bar
I[-1] = (V0 - B0 * L * v[-1]) / R_bar

# graficar v(t) y marcar v_inf
plt.figure(figsize=(8,5))
plt.plot(time2, v, label='v(t) numérica')
plt.hlines(v_inf, time2[0], time2[-1], colors='k', linestyles='--', label=f'v_inf teórica = {v_inf:.1f} m/s')
plt.scatter(time2[-1], v_inf, color='k', zorder=6)
plt.annotate(f"v_inf = {v_inf:.1f} m/s", xy=(time2[-1], v_inf),
             xytext=(time2[-1]*0.2, v_inf*0.6),
             arrowprops=dict(arrowstyle="->"), fontsize=10)
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.title('Velocidad de la barra en función del tiempo (marcado v_inf)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# graficar I(t) y marcar I_inf (cero)
plt.figure(figsize=(8,5))
plt.plot(time2, I*1e3, label='I(t) numérica')
plt.hlines(0, time2[0], time2[-1], colors='k', linestyles=':', label='I_inf = 0 A')
plt.annotate("I_inf = 0", xy=(time2[-1]*0.6, 0.0), xytext=(time2[-1]*0.4, 0.5),
             arrowprops=dict(arrowstyle="->"), fontsize=10)
plt.xlabel('t [s]')
plt.ylabel('I [mA]')
plt.title('Corriente en función del tiempo (barra móvil)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
