import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def fcbiseccion_NR(f, df, inter, tol, max_iter_NR=100):
    # Defino intervalo
    a, b = inter

    # Chequeo si sirve el intervalo 
    if f(a) * f(b) >= 0:
        print("No sirve el intervalo")
        return None, None, None

    # Bisección 
    for i in range(5):  # 5 pasos de bisección
        x0 = (a + b) / 2
        if f(a) * f(x0) < 0:
            b = x0
        else:
            a = x0

    inter = [a, b]
    x0 = (a + b) / 2  # Punto inicial para Newton-Raphson
    print(f"Intervalo reducido con bisección: {inter}")

    # Newton-Raphson 
    i=0
    while True:
        i+=1
        if df(x0) == 0: #esto seria el bolsano?
            print("Derivada nula, no se puede continuar")
            return None, None, inter

        x1 = x0 - f(x0) / df(x0)
        
        if abs(x1 - x0) < tol or abs(f(x1)) < tol:
            print(f"Newton-Raphson convergió en {i} iteraciones")
            return x1, abs(f(x1)), inter
        if i >= max_iter_NR:
            print("No convergió en el máximo de iteraciones")
            return None, None, inter
        x0 = x1
    