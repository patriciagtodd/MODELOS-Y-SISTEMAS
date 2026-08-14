# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
R = 1e3        # ohm
C1 = 10e-6      # F
C2 = 1e-6      # F
Q1_0 = 1e-6    # C (C1 cargado con 1 µC)
Q2_0 = 0       # C (C2 descargado)
step = 1e-4
time = np.arange(0, 0.01, step)

# Inicialización
Q1 = [Q1_0]
Q2 = [Q2_0]

# Ecuación diferencial (Euler)
for t in time[1:]:
    dQ1 = (-(Q1[-1]/R)*(1/C1 + 1/C2) + (Q1_0/(R*C2))) * step
    Q1_new = Q1[-1] + dQ1
    Q2_new = Q1_0 - Q1_new  # conservación de la carga
    Q1.append(Q1_new)
    Q2.append(Q2_new)

# Conversión a arrays
Q1 = np.array(Q1)
Q2 = np.array(Q2)

# Gráficos
plt.figure(figsize=(8,5))
plt.plot(time, Q1*1e6, label="Q1(t) [µC]", color='orange')
plt.plot(time, Q2*1e6, label="Q2(t) [µC]", color='blue')
plt.title("Evolución de las cargas Q1 y Q2 en función del tiempo", fontsize=14)
plt.xlabel("Tiempo [s]", fontsize=12)
plt.ylabel("Carga [µC]", fontsize=12)
plt.grid(True)
plt.legend()
plt.show()

# Agrego corriente
I = -np.diff(Q1)/step
plt.figure(figsize=(10,5))
plt.plot(time[:-1], I, label="Corriente [A]")
plt.title("Evolución de la corriente en función del tiempo", fontsize=14)
plt.xlabel("Tiempo [s]", fontsize=12)
plt.ylabel("Corrriente [µC]", fontsize=12)
plt.legend()
plt.show()
