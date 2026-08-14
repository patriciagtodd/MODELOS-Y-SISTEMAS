import funciones  
import numpy as np 
import matplotlib.pyplot as plt

# EDO original
f = lambda t, y: 2 - np.sqrt(np.abs(1 - np.exp(np.sin(t)))) * y

ti = 0
tf = 10
u0 = -1
n = 30
h = (tf - ti)/n

# Solución de u(t): Heun
T_h, U_h = funciones.fc_Heun(f, ti, tf, u0, h)
plt.plot(T_h, U_h, 'r', label="u(t) con Heun")

# Ajuste polinomial u ≈ p(t)
grado = 5
coef = np.polyfit(T_h, U_h, grado)
u = lambda t: np.polyval(coef, t)

# Resolver y' = u(t)
y0 = 4
g = lambda t, Y: u(t)

T_y, Y_y = funciones.fc_Heun(g, ti, tf, y0, h)
plt.plot(T_y, Y_y, 'c', label="y(t)")

# Integrar u(t) para obtener y(10)
Iy = funciones.fc_simpson_compuesto_vector(U_h, ti, tf, h)
y10 = y0 + Iy
print("y(10) =", y10)

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
