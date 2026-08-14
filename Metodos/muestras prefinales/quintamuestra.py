import numpy as np
import matplotlib.pyplot as plt

# Función y su derivada
f = lambda x: np.sqrt(x*x + 1) + (1/3)*x**3 - 1
f_derivada = lambda x: (x / np.sqrt(x*x + 1)) + x*x

# Parámetros
tolerancia = 1e-6

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

# Ejecuto Newton-Raphson
raiz, error = fc_NR(f, f_derivada, tolerancia)

print(f"Raíz aproximada: {raiz}")
print(f"Error estimado: {error}")

# Gráfico
x = np.linspace(-5, 5, 200)
y1 = f(x)
plt.plot(x, y1, 'r', label="f(x)")

# Ejes coordenados
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')

# Marca la raíz encontrada
if raiz is not None:
    plt.plot(raiz, f(raiz), 'bo', label='Raíz')

plt.legend()
plt.grid(True)
plt.show()
