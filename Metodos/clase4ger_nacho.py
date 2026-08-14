import numpy as np
import matplotlib.pyplot as plt
#import clase4ger
from clase4ger import fcbiseccion

funcion = lambda x : x + 1.5*np.exp(-(np. sin (2*x)))
tolerancia = 0.5/(np. e)
intervalo = [-2,10]

mensaje = fcbiseccion(intervalo,funcion,tolerancia)
print (mensaje)