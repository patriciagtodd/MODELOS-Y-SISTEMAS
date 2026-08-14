import numpy as np
import matplotlib.pyplot as plt

# n+1 nodos
# n divisiones
# t0 es dato
# y0 dato
# voy de t[0] a t[n] 

## DATOS ##
t0 = 0
y0 = 10
tf = 1
n = 4
f = lambda t,y : -0.2*y 
h = (tf - t0)/n
## FUNCION ##

def MetodoEuler (f, t0, y0, tf, h, n):
    M = int(( tf - t0)/ h)
    y = np.zeros(n + 1) ## en el cuaderno dice len(t)
    t = np.arange(t0,tf + h, h)
    t[0] = t0
    y[0] = y0
    
    for k in np.arange(0, n):
        y[k +1] = y[k] + h*f(t[k],y[k])

    return t, y

t, y = MetodoEuler(f, t0, y0, tf, h, n)
plt.plot(t,y)
plt.show()
print("Resultado:", y)
