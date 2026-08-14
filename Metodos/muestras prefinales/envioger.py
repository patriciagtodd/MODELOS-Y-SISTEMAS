import numpy as np 
import matplotlib.pyplot as plt
from funciones import fcbiseccion

'''#EJERCICIO 1 
# Función y su derivada
f1 = lambda x: np.exp(2*x - x**2 -(3/4)) -(2/3)*x**2 + 3*x -4

#grafico
x = np.linspace(0, 2, 100)
y1 = f1(x)
plt.plot(x, y1, 'r', label="f(x)")

# Ejes coordenados
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')


plt.legend()
plt.grid(True)
plt.show()

# EN EL INTERVALO [0,3] hago que haga igual bisecciona ver que pasa
inter = [0,2]
tol = 1e-4 #esto es 0,0001 y al pedir menor que esto ya son las 4
print(tol)
# VEO QUE ONDA LA BISECCION
print("entro a biseccion para demostarr que lo rechaza")
x0 = fcbiseccion(f1, inter, tol) 
#NO FUNCIONA LA BISEECION PORQUE NO cumple bolzano

#uso N-R aunque creo que voy a necesitar n-r modificado
d1f1 =  lambda x: (2 - 2*x)*(np.exp(2*x - x**2 -(3/4))) - (4/3)*x + 3
d2f1 = lambda x: ((-2)*(np.exp(2*x - x**2 -(3/4))) +((2 - 2*x)**2)*(np.exp(2*x - x**2 -(3/4)))) - (4/3)

from funciones import fc_NR
x0, error = fc_NR(f1, d1f1, tol, 5, 1.25)
print(f"La raiz aproximada con n-r simple es {x0} con un error de {error}")

from funciones import fc_NR_mod
x0, error = fc_NR_mod(f1, d1f1, d2f1, tol, 5, 1.25)
print(f"La raiz aproximada con n-r modificado es {x0} con un error de {error}")

'''

#=================================================================================
#ejercicio 2

f = lambda t, y: (np.sin(y**2 + t*y - t**2))/(y**2 - t**2)
ti = 0
tf = 5
y0 = np.sqrt(np.pi)
N = 10 
h = (tf - ti)/N
from funciones import fc_euler
T_e, Y_e = fc_euler(f,ti, tf, y0, h)

#grafico euler
plt.plot(T_e, Y_e, 'r', label="y(t), euler")


from funciones import fc_Heun
T_h, Y_h = fc_Heun(f, ti, tf, y0, h)
# grafico heun
plt.plot(T_h, Y_h, 'm', label="y(t), heun")

from  funciones import fc_RK4
T_R, Y_R = fc_RK4(f, ti, tf, y0, h)
# grafico heun
plt.plot(T_R, Y_R, 'c', label="y(t), Runge-Kutta 4")

#Errores
errorE  = abs(Y_R - Y_e)
plt.plot(T_R, errorE, 'r--', label="error euler respesto rK4 ")

errorH  = abs(Y_R - Y_h)
plt.plot(T_R, errorH, 'c--', label="error heun respesto rK4 ")

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Soluciónes')
plt.legend()
plt.grid(True)
plt.show()

N2 = 60 
h2 = (tf - ti)/N2

#Euler
T_e2, Y_e2 = fc_euler(f,ti, tf, y0, h2)
plt.plot(T_e2, Y_e2, 'r', label="y(t), euler")

#Heun
T_h2, Y_h2 = fc_Heun(f, ti, tf, y0, h2)
plt.plot(T_h2, Y_h2, 'm', label="y(t), heun")

# Runge-Kutta 4 
T_R2, Y_R2 = fc_RK4(f, ti, tf, y0, h2)
plt.plot(T_R2, Y_R2, 'c', label="y(t), Runge-Kutta 4")

#Errores
errorE  = abs(Y_R2 - Y_e2)
plt.plot(T_R2, errorE, 'r--', label="error euler respesto rK4 ")

errorH  = abs(Y_R2 - Y_h2)
plt.plot(T_R2, errorH, 'c--', label="error heun respesto rK4 ")

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Soluciónes')
plt.legend()
plt.grid(True)
plt.show()


# simpson tenindo ya nodos 
from funciones import fc_simpson_compuesto_vector
# CON N = 10
I_e = fc_simpson_compuesto_vector(Y_e, ti, tf, h)
print (f"Integral Simpson EULER (N=10): {I_e}")

I_h = fc_simpson_compuesto_vector(Y_h, ti, tf, h)
print (f"Integral Simpson HEUN (N=10): {I_h}")

#AHORA CON N= 60
I_e2 = fc_simpson_compuesto_vector(Y_e2, ti, tf, h2)
print (f"Integral Simpson EULER (N=60): {I_e2}")
Ih_2 = fc_simpson_compuesto_vector(Y_h2, ti, tf, h2)
print (f"Integral Simpson HEUN (N=60): {Ih_2}")
