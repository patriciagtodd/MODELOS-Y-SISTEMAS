import numpy as np 
from TP1_metodos_Gonzaleztodd import fcbiseccion_NR
import matplotlib.pyplot as plt

# Defino la función y su derivada
funcion  = lambda x: (3/5)*np.exp((15/7)*np.cos(3*x)) - 2*x
funcion_derivada = lambda x: (-27/7)*np.exp((15/7)*np.cos(3*x))*np.sin(3*x) - 2

# Parámetros 1ra raíz
tolerancia = 5e-6
intervalo = [-9, 1]

'''# Parámetros 2da raíz
tolerancia = 5e-6
intervalo = [1, 2]'''

# Ejecuto bisección + Newton-Raphson
raiz, error, inter = fcbiseccion_NR(funcion, funcion_derivada, intervalo, tolerancia)

print(f"Raíz aproximada: {raiz}")
print(f"Error estimado: {error}")

# gráfico
x = np.linspace(-5, 5, 100)
y1 = funcion(x)
plt.plot(x, y1, 'r', label="f(x)")  # r=rojo

# Ejes coordenados
plt.axhline(0, color='k', linestyle='--')  # eje x
plt.axvline(0, color='k', linestyle='--')  # eje y

# Opcional: leyenda y grilla
plt.legend()
plt.grid(True)

# agrega punto en la raiz 
plt.plot(raiz, funcion(raiz), 'bo', label='Raíz')  # punto azul en la raíz

plt.show()
