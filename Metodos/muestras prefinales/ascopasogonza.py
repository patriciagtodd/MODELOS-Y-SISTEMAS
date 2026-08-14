import funciones  
import numpy as np 
import matplotlib.pyplot as plt

f = lambda t, u: 2 - (np.sqrt(np.abs(1- np.exp(np.sin(t)))))*u
ti = 0
tf = 10
u0 = -1
n = 30
h = (tf - ti)/n

T_e, U_e = funciones.fc_euler(f,ti, tf, u0, h)

#grafico euler 31 nodos
plt.plot(T_e, U_e, 'g', label="u(t), euler 31 nodos")

T_h, U_h = funciones.fc_Heun(f,ti,tf,u0,h)
plt.plot(T_h, U_h, 'r', label="u(t), Heun 31 nodos")

# f(b)-f(a) = integral(f'dx)|[a,b] => u' = f =  y'' -> u = integral(f) = y'
# y10 = integral f + y0 = integral U + y0

y0 = 4
coef = np.polyfit(T_h, U_h, len(T_h)-1)
u = lambda t: np.polyval(coef, t)
g = lambda t , Y: u(t)

#resuelvo la diferencial de nuevo 
t_h, y_h = funciones.fc_Heun(g,ti,tf,y0,h) 
plt.plot(t_h, y_h, 'c', label="y(t), Heun 31 nodos")

# integro
Iy = funciones.fc_simpson_compuesto_vector(U_h, ti, tf, h)
y10 = Iy + y0 
print(y10)

plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.show()


