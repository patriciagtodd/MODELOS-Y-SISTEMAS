import funciones  
import numpy as np 
import matplotlib.pyplot as plt

#====================================================================================
# EJERCICIO 1
'''f1 = lambda x: 2*np.log(4*x - 7) + x**2 - (15/2)*x + (209/16) - 2*np.log(4)

#redefinicion para puntoo semilla
g = lambda x: 2*np.log(4*x - 7) + x**2 - (15/2)*x + (209/16) - 2*np.log(4) - x

# Gráfico f
x = np.linspace(2, 10, 1000)
y1 = f1(x)
plt.plot(x, y1, 'r', label="f(x)")

#Grafico g
y2 = g(x)
plt.plot(x, y2, 'c', label="g(x)")

# Ejes
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')

plt.legend()
plt.grid(True)
plt.show()

# es una funcion que no es  tan dificil de derivar pero aunque quiera usar newton el error puede ser terrrible
# porque la derivada cerca de la raiz tiende a cero

# Bisección: intervalo válido
tol = 1e-4
inter = [2, 3]

x0 = funciones.biseccion(f1, inter, tol)

tol2 = 1e-4
inter = [4,6]

xsemila = funciones.biseccion(g, inter, tol2)
'''

#===============================================================================
#EJERCICIO 2
'''
f = lambda x: 8*(np.exp(np.sin(x)))-x
#grafico para ver el intervalo de biseccion o si es mejor newton
x = np.linspace(0, 10, 1000)
y3 = F2(x)
plt.plot(x, y3, 'r', label="f(x)")

# Ejes
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')

plt.legend()
plt.grid(True)
plt.show()

# Bisección: intervalo válido
tol = 1e-4
inter = [3, 4] #recuerda que esto realmente es un vector no un arange intervalo

x0 = funciones.biseccion(f, inter, tol)

#ya encontrado el x0 integro
# me piden 11 nodos por tanto 10 pasos o subintervalos asi que puedo hacer 
# simpson compuesto y trapecios compuestos
n = 10
IT = funciones.TrapreciosCompuesto(f, inter[0], inter[1], n)
IS = funciones.SimpsonCompuesto(f, inter[0], inter[1], n)
'''

#Prueba funcion spline

xi = np.array([0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5, 6.0, 7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3])
fi = np.array([1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1, 2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 0.25])

xfino, yfino = funciones.splinecubico(xi, fi)

