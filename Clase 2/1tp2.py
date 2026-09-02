import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Ejercicio 1: y' = y - 1, y(0) = 2, en [0,7] ---

# Opción A: solución analítica 
y0 = 2
a = 1
b = -1 
C = y0 + (b/a)
tmax = 7
t_analitico = np.linspace(0, tmax, 200)
y_analitico = (-b/a) + C*np.exp(a*t_analitico)

# Opción B: solución numérica con solve_ivp
def f(t, y):
    return y - 1   # la ecuación diferencial, no la solución

sol = solve_ivp(f, [0, tmax], [2], method='RK23', t_eval=np.linspace(0, tmax, 200))

# Graficar ambas
plt.plot(t_analitico, y_analitico, label='Analítica', linewidth=2)
plt.plot(sol.t, sol.y[0], '--', label='Numérica (RK23)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Ejercicio 1: y' = y - 1")
plt.legend()
plt.grid(True)
plt.show()

# %%# --- Ejercicio 2: y' = 0.2y, y(0) = 30000 , en [0,10]horas ---

# Opción A: solución analítica 
y0 = 30000
a = 0.2
b = 0 
C = y0 + (b/a)
tmax = 10
t_analitico = np.linspace(0, tmax, 200)
y_analitico = (-b/a) + C*np.exp(a*t_analitico)

# Opción B: solución numérica con solve_ivp
def f(t, y):
    return (0.2)*y   # la ecuación diferencial, no la solución

sol = solve_ivp(f, [0, tmax], [y0], method='RK23', t_eval=np.linspace(0, tmax, 200))

# Graficar ambas
plt.plot(t_analitico, y_analitico, label= f"Analítica, y(10) = {y_analitico[tmax]}", linewidth=2)
plt.plot(sol.t, sol.y[0], '--', label= "Numérica (RK23), y(10) = ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title("Ejercicio 2: y' = 0.2y, y(0) = 30000 , en [0,10]horas")
plt.legend()
plt.grid(True)
plt.show()

# %%

