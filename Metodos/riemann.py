import numpy as np
a = 0
b = 1
f = lambda x : np.exp(x)
N = 10
I_exacta = np.exp(1) - 1  # integral exacta de e^x entre 0 y 1

print(f"Valor exacto: {I_exacta:.10f}\n")

for N in range(10, 101, 10):
    h = (b-a)/N

    pm = np.arange(a+h/2,b,h) #vector de los puntos medios
    ei = np.arange(a,b,h) #vector de los extremos izquierdos

    y_pm = f(pm)
    y_ei = f(ei)

    integrRiem_pm = sum(y_pm)*h
    integrRiemie_ei = sum(y_ei)*h
    print (f"N = {N}")
    print("punto medio") 
    print(integrRiem_pm)

    print("extremos izquierdos")
    print(integrRiemie_ei)
## error maximo
err = 1e-4

# Derivada: f'(x) = e^x → M = e^b = e^1
M = np.exp(1)

# Fórmula de la cota de error del método de Riemann (extremo izquierdo):
# E_T ≤ (M/2) * (b - a)^2 / N
# Se pide E_T < eps → N ≥ M*(b - a)^2 / (2*eps)
N_min = (M * (b - a)**2 / (2 * err))

print(f"Para que el error sea menor que {err}, se necesita N >= {N_min}")