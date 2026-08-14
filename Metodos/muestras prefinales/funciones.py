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
def fc_NR(f, df, tol, max_iter_NR=100):
    x0 = 0.5
    x0 = f(x0) # punto inicial en la recta y=x
    i = 0
    while True:
        i += 1
        if df(x0) == 0:
            print("Derivada nula, no se puede continuar")
            return None, None

        x1 = x0 - f(x0) / df(x0)
        
        if abs(x1 - x0) < tol or abs(f(x1)) < tol:
            print(f"Newton-Raphson convergió en {i} iteraciones")
            return x1, abs(f(x1))
        if i >= max_iter_NR:
            print("No convergió en el máximo de iteraciones")
            return None, None
        
        x0 = x1
    return x0

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
def fc_TrapreciosCompuesto(f, a, b, h):
    S = 0
    for k in np.arange(a + h, b, h):
        S = S + f(k) # sumatoria intermedia
    I = h * (((f(a) + f(b)) / 2) + S)
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

# SIMPSON COMPUESTO
def fc_SimpsonCompuesto(f,a,b,h):
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

# ==============================================================================
#EULER
def fc_euler(f, t0, tf, y0, N):
    """
    Resuelve una EDO de primer orden usando el método de Euler.

    Parámetros:
    -----------
    f : función
        Derivada de y, debe aceptar (t, y). Ej: f = lambda t, y: 2*t*y
    t0 : float
        Tiempo inicial
    tf : float
        Tiempo final
    y0 : float
        Valor inicial de y(t0)
    N : int
        Número de pasos (habrá N+1 nodos)

    Retorna:
    --------
    T : ndarray
        Vector con los tiempos (nodos)
    Y : ndarray
        Vector con los valores aproximados de y(t)
    """
    h = (tf - t0) / N
    T = np.linspace(t0, tf, N + 1)
    Y = np.zeros(N + 1)
    Y[0] = y0

    for k in range(N):
        Y[k + 1] = Y[k] + h * f(T[k], Y[k])

    return T, Y
