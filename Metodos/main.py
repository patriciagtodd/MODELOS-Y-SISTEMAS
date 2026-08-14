import numpy as np 
from biseccion import fcbiseccion
funcion  = lambda x: x + np.exp(-(np.sin(2*x)))
tolerancia=0.5e-1
intervalo = [-2, 10]
resultado = fcbiseccion (funcion, intervalo, tolerancia)
