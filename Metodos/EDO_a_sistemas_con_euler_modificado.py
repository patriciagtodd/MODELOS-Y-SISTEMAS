import numpy as np
import matplotlib.pyplot as plt
'''
los dos puntitos sirven para recorrer
f esta definida como una funcion que va de r3 a r2
'''

f = lambda t, y, z: np.array([-np.sin(3*y)+3*t, z]);
t0 = 0
tf = 2
N = 20
h = (tf-t0)/N
t = np.arange(t0, tf + h , h)
y = np.zeros((2, N + 1))
y [0,0] = 2 #y0
y [1,0] = 1 # z0

#EULER
for k in np.arange(0,N):
    y [:, k +1] = y[:, k] + h * f(t[k], y[0, k], y[1, k]);

print(y)
plt.plot (t, y[0,:], 'r')
plt.plot(t, y[1, :],'b')
plt.show()