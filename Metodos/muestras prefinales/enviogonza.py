import funciones  
import numpy as np 
import matplotlib.pyplot as plt

#===========================================================================
#EJERCICIO 1
'''
f = lambda  t, y : np.sin((1/3)*t**2 + y -  3)
ti = 0
tf = 10
y0 = 3
N = 16
# tengo N = 16nodos entonces n = 15 pasos por tanto no puedo usar simpson compuesto, solo puedo usar trapecios o combianr
h = (tf - ti)/(N-1) # aca van los pasos

T_e, Y_e = funciones.fc_euler(f,ti, tf, y0, h)

#grafico euler 15 nodos
plt.plot(T_e, Y_e, 'c', label="y(t), euler 16 nodos")

I = funciones.fc_TrapreciosCompuesto_vector(Y_e, ti, tf, h)
print(f"La integral es {I}")

N2 = 100
h2 = (tf - ti)/(N2-1) # aca van los pasos

T_e2, Y_e2 = funciones.fc_euler(f,ti, tf, y0, h2)

#grafico euler 100 nodos
plt.plot(T_e2, Y_e2, 'm', label="y(t), euler 100 nodos")

I = funciones.fc_TrapreciosCompuesto_vector(Y_e2, ti, tf, h2)
print(f"La integral es {I}")

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Soluciónes')
plt.legend()
plt.grid(True)
plt.show()'''

#===========================================================================
#EJERCICIO 1

X = [-1,1,2,2.5,3.5,5,5.5,6,7]
Y = [3,3,2,1,2,-1,-2,-1,-0.5]

coef = np.polyfit(X, Y, (len(X)-1))

x_fino = np.arange(min(X), max(X), 0.1) # puede ser esto o np.linspace(min(X), max(X), 1000)
y_fino = np.polyval(coef,x_fino)

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.plot(X, Y, 'ro', label='Datos')
plt.plot(x_fino, y_fino, 'b-', label='Ajuste polinomial')
plt.legend()
plt.grid(True)
plt.show()

#viendo el grafico se que me sirve el intervalo [0,1] y me sirve biseccion porque la pendientes espronunciada

f = lambda x: np.polyval(coef,x)

inter = [0, 2]
tol = 1e-3 #esto es 0,001 y al pedir menor que esto ya son las 4

x0 = funciones.fcbiseccion(f, inter, tol)
if( x0 != None):

    # gráfico
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.plot(X, Y, 'ro', label='Datos')
    plt.plot(x_fino, y_fino, 'b-', label='Ajuste polinomial')
    plt.plot(x0, f(x0), 'bo', label='Raíz')
    plt.legend()
    plt.grid(True)
    plt.show()




