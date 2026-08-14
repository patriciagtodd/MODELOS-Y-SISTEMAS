import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#BISECCION
def fcbiseccion(f, inter, tol):
    a = inter[0]
    b = inter[1]

    if f(a) * f(b) < 0:
        x0 = (a + b) / 2
        r = (b - a) / 2

        while r > tol:
            if f(a) * f(x0) < 0:
                b = x0
            else:
                a = x0
            x0 = (a + b) / 2
            r = (b - a) / 2

        print(f"La raíz es: {x0}")
        print(f"El error es: {r}")

    else:
        print("No sirve el intervalo")
        return None

    # gráfico
    x = np.linspace(inter[0], inter[1], 200)
    y1 = f(x)
    plt.plot(x, y1, 'r', label="f(x)")
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.plot(x0, f(x0), 'bo', label='Raíz')
    plt.legend()
    plt.grid(True)
    plt.show()

    return x0

# ==============================================================================
#NEWTON-RAPHSON
def fc_NR(f, df, tol, max_iter_NR, x0):
    i = 0
    while True:
        i += 1
        if df(x0) == 0:
            print("Derivada nula, no se puede continuar")
            return None, None

        x1 = x0 - f(x0) / df(x0)
        
        if abs(x1 - x0) < tol:
            print(f"Newton-Raphson convergió en {i} iteraciones")
            return x1, abs(x1 - x0)
        if i == max_iter_NR:
            print("No convergió en el máximo de iteraciones => envio el aproximado")
            return x1, abs(x1 - x0)
        x0 = x1
    return None, None

def fc_NR_mod(f, df, d2f, tol, max_iter, x0):
    i = 0
    while i < max_iter:
        i += 1
        denom = (df(x0))**2 - f(x0)*d2f(x0)
        if denom == 0:
            print("Denominador nulo, se detiene.")
            return None, None
        x1 = x0 - ((f(x0)*df(x0))/denom)
        if abs(x1 - x0) < tol:
            print(f"Newton-Raphson modificado convergió en {i} iteraciones")
            return x1, abs(x1-x0)
        x0 = x1
    print("No convergió en el máximo de iteraciones")
    return None, None

# ==============================================================================
#INTEGRAL DE RIEMAN
def fc_IntegrRieman(f, a, b, N, h):
    pm = np.arange(a+h/2,b,h) #vector de los puntos medios
    y=f(pm)
    IntegrRiem=sum(y)*h
    print(IntegrRiem)
    return IntegrRiem

# ==============================================================================
#TRAPECIO
'''It = np.trapz(ym, xm)
print(f"5.2 metodo de trapecios:       {It:.6f}")'''

#TRAPECIOS COMPUESTOS 
def fc_TrapreciosCompuesto(f, a, b, n):
    x = np.linspace(a, b, n+1)  # n+1 nodos → n intervalos
    y = f(x)
    h = (b - a)/n
    I = h * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])
    return I

# ==============================================================================
#SIMPSON

def fc_simpson(f, a, b, N):
    """
    Aproxima la integral de f(x) entre a y b
    usando el método de Simpson 1/3 compuesto.

    Parámetros:
    -----------
    f : función
        La función a integrar, por ejemplo f = lambda x: np.sin(x)
    a : float
        Límite inferior de integración
    b : float
        Límite superior de integración
    N : int
        Número de subintervalos (debe ser par)

    Retorna:
    --------
    float : aproximación de la integral
    """
    if N % 2 != 0:
        raise ValueError("N debe ser par para aplicar el método de Simpson.")

    h = (b - a) / N  # ancho del subintervalo
    x = np.linspace(a, b, N + 1)  # nodos
    y = f(x)  # valores de la función en los nodos

    # Suma según la fórmula compuesta
    S = y[0] + y[-1] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2])
    I = (h / 3) * S

    return I

# SIMPSON COMPUESTO TENIEDO FUNCION
def fc_SimpsonCompuesto(f,a,b,n):
    h = (b - a) / n
    N = int((b- a) / h)
    if N % 2 != 0: # Si Nno es par
        raise ValueError("N debe ser par")
    SP = 0 # términos pares
    SI = 0 # términos impares
    for k in np.arange(a + 2*h, b, 2*h):
        SP = SP + f(k)
    for k in np.arange(a + h, b, 2*h):
        SI = SI + f(k)
    I = (h / 3) * (f(a) + f(b) + 2*SP + 4*SI)
    return I

def fc_simpson_compuesto_vector(Y, ti, tf, h):
    # simpson tenindo ya nodos 
    SP = np.sum(Y[2:-2:2]) 
    SI = np.sum(Y[1:-1:2])

    I = (h / 3) * (Y[0] + Y[-1] + 2 * SP + 4 * SI)

    return I

def fc_SimpsonTrapecioMixto(f, a, b, n):
    # n = cantidad de subintervalos (en este caso 7)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson hasta el anteúltimo nodo (6 subintervalos → 7 nodos)
    I_simpson = (h / 3) * (y[0] + 4 * np.sum(y[1:n-1:2]) + 2 * np.sum(y[2:n-2:2]) + y[n-1])

    # Último subintervalo (trapecio)
    I_trap = (h / 2) * (y[n] + y[n])

    return (I_simpson + I_trap), I_simpson, I_trap

# ==============================================================================
#EULER
def fc_euler(f, t0, tf, y0, h):
    '''
    Resuelve una EDO de primer orden usando el método de Euler.'''
    M = int((tf - t0) / h)
    T = np.arange(t0,tf + h, h)
    Y = np.zeros(len(T))
    Y[0] = y0

    for k in range(M):
        Y[k + 1] = Y[k] + h * f(T[k], Y[k])

    return T, Y
#===================================================================
# Heum
def fc_Heun(f, t0, tf, y0, h):
    M = int((tf- t0) / h)
    t = np.arange(t0, tf + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for k in range(M):
        K1 = f(t[k], y[k])
        K2 = f(t[k] + h, y[k] + h * K1)
        y[k+1] = y[k] + (h / 2) * (K1 + K2)
    return t, y

#===================================================================
# runge-Kutta 4
def fc_RK4(f, t0, tf, y0, h):
    M = int((tf- t0) / h)
    t = np.arange(t0, tf + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for k in range(M):
        K1 = f(t[k], y[k])
        K2 = f(t[k] + h/2, y[k] + h * K1/2)
        K3 = f(t[k] + h/2, y[k] + h * K2/2)
        K4 = f(t[k] + h, y[k] + h * K3)
        y[k+1] = y[k] + h * (K1 +2*K2+2*K3+K4)/6
    return t, y