import numpy as np
import matplotlib.pyplot as plt

def fcbiseccion(inter,f,tol):
    a=inter[0]
    b=inter[1]
    x = np.linspace (-10,10)
    y1 = f(x)  
    plt. plot(x,y1,'r')
    plt.show()  
    if f(a)*f(b) < 0:
        medio = (a+b)/2
        r = (b-a)/2
        while r > tol :
            if f(a)*f(medio) < 0:
                b = medio
            else:
                a = medio
            medio = (a+b)/2
            r = (b-a)/2

    print ("La raiz se aproxima a: ",medio)
    print ("El error es: ",r)

funcion = lambda x : x + 1.5*np.exp(-(np. sin (2*x)))
tolerancia = 0.5/(np. e)
intervalo = [-2,10]
fcbiseccion (intervalo,funcion,tolerancia)