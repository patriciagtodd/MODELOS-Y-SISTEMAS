import numpy as np
import matplotlib.pyplot as plt

f = lambda t, y, z: np.array([-0.2*np.cos(3*y), .1*y+2*np.sin(2*z)]) 
# es una funcion tipo transformacion que va de r3 a r2b
t0 = 0
tf = 2
N = 20
h = (tf-t0)/N
t = np.arange(t0, tf + h , h)
y = np.zeros((2, N + 1))
y [0,0] = 2 #y0
y [1,0] = 1 # z0

for k in np.arange(0,N):
    y [:, k +1] = y[:, k] + h * f(t[k], y[0, k], y[1, k]);

print(y)
plt.plot (t, y[0,:], 'r')
plt.plot(t, y[1, :],'b')
plt.show()


# en el errorhay que hacer un vector as donde resta de de3 co  el de 4, le cambia la escala al grafico y veo como cambia