
# -*- coding: utf-8 -*-
"""
Trabajo Final - Modelos y Sistemas
Modelo reducido de combustion

Parte 1: Modelo no lineal
Parte 2: Puntos de equilibrio
Parte 3: Linealizacion y estabilidad
"""

# --------------------------------------------------
# IMPORTACION DE LIBRERIAS
# --------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp


# --------------------------------------------------
# 1. PARAMETROS DEL MODELO
# --------------------------------------------------

k0 = 10**4       # Constante de velocidad de reaccion [1/s]
ER = 5000        # Cociente E/R [K]
q = 100          # Parametro de generacion de calor [K]
h = 0.5          # Coeficiente de intercambio termico [1/s]
Ta = 300         # Temperatura ambiente [K]
alfa = 0.1       # Coeficiente de intercambio termico interno [1/s]
beta = 10        # Acoplamiento con el combustible [K/s]


# --------------------------------------------------
# 2. CONDICIONES INICIALES
# --------------------------------------------------

# Condiciones iniciales proporcionadas por la catedra.

Y0 = 1
T0 = 300
z0 = 300

# Agrupamos las tres variables en un vector.
# El sistema tiene tres variables de estado.

x0 = [Y0, T0, z0]


# --------------------------------------------------
# 3. INTERVALO DE SIMULACION
# --------------------------------------------------

t_inicio = 0
t_final = 2
N = 5000

# Creamos 5000 instantes entre 0 y 2 segundos.
# Como se incluyen ambos extremos, existen N-1 intervalos.

t_eval = np.linspace(t_inicio, t_final, N)


# --------------------------------------------------
# 4. MODELO NO LINEAL
# --------------------------------------------------

def modelo_nolineal(t, x):

    # Separamos las tres variables de estado.
    Y = x[0]
    T = x[1]
    z = x[2]

    # Factor exponencial de Arrhenius.
    # Representa la dependencia de la reaccion
    # quimica con la temperatura.

    factor = np.exp(-ER/T)

    # Ecuacion 1: consumo de combustible.
    dYdt = -k0 * factor * Y

    # Ecuacion 2: generacion y perdida de calor.
    dTdt = q * k0 * factor * Y - h * (T - Ta)

    # Ecuacion 3: evolucion de la variable termica interna.
    dzdt = alfa * (T - z) + beta * Y

    # Devolvemos las derivadas de los tres estados.
    return [dYdt, dTdt, dzdt]


# %%
# --------------------------------------------------
# 5. SIMULACION DEL MODELO NO LINEAL
# --------------------------------------------------

# Resolvemos las tres ecuaciones diferenciales
# simultaneamente utilizando solve_ivp.
#
# RK45 es un metodo Runge-Kutta de paso adaptativo.
# t_eval indica los instantes donde queremos
# obtener los resultados.

sol = solve_ivp(
    modelo_nolineal,
    [t_inicio, t_final],
    x0,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10
)

# Comprobamos que la simulacion haya sido exitosa.
print(sol.message)

# Extraemos las soluciones de las tres variables.
tt = sol.t
YY = sol.y[0, :]
TT = sol.y[1, :]
zz = sol.y[2, :]


# --------------------------------------------------
# 6. SOLUCIONES TEMPORALES DEL MODELO NO LINEAL
# --------------------------------------------------

plt.figure(figsize=(10, 8))

# Concentracion de combustible.
plt.subplot(3, 1, 1)
plt.plot(tt, YY, color='blue')
plt.ylabel('Y(t)')
plt.title('Modelo no lineal - Soluciones temporales')
plt.grid()

# Temperatura.
plt.subplot(3, 1, 2)
plt.plot(tt, TT, color='red')
plt.ylabel('T(t) [K]')
plt.grid()

# Variable termica interna.
plt.subplot(3, 1, 3)
plt.plot(tt, zz, color='green')
plt.xlabel('Tiempo [s]')
plt.ylabel('z(t) [K]')
plt.grid()

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------
# 7. CALCULO DE LOS PUNTOS DE EQUILIBRIO
# --------------------------------------------------

# Seguimos el procedimiento utilizado por la profesora:
#
# 1. Definir las variables simbolicas.
# 2. Igualar las tres ecuaciones a cero.
# 3. Resolver el sistema mediante solve.
# 4. Convertir las soluciones a valores numericos.

# Definimos las variables simbolicas.
Y_s, T_s, z_s = sp.symbols('Y T z', real=True)

# Escribimos las ecuaciones del modelo no lineal.
# Usamos exp() de SymPy para realizar calculos simbolicos.

f1 = -k0 * sp.exp(-ER/T_s) * Y_s

f2 = q * k0 * sp.exp(-ER/T_s) * Y_s - h * (T_s - Ta)

f3 = alfa * (T_s - z_s) + beta * Y_s

# Igualamos todas las derivadas a cero.
# Los puntos de equilibrio cumplen:
# dY/dt = 0, dT/dt = 0 y dz/dt = 0.

eqs = [
    sp.Eq(f1, 0),
    sp.Eq(f2, 0),
    sp.Eq(f3, 0)
]

# Resolvemos el sistema de ecuaciones simultaneamente.
# dict=True devuelve cada solucion como un diccionario.

S = sp.solve(eqs, [Y_s, T_s, z_s], dict=True)

print("\nPuntos de equilibrio simbolicos:")
print(S)

# Convertimos los resultados a valores numericos.
# Cada fila de Xcrit representa un punto de equilibrio.

Xcrit = np.array([
    [float(s[Y_s]), float(s[T_s]), float(s[z_s])]
    for s in S
])

print("\nPuntos de equilibrio numericos:")
print(Xcrit)

# Seleccionamos el equilibrio para analizarlo.
xe = Xcrit[0, :]

# Comprobamos el resultado evaluando las derivadas.
# En un equilibrio, las tres deben valer cero.

derivadas_equilibrio = modelo_nolineal(0, xe)

print("\nDerivadas en el equilibrio:")
print(derivadas_equilibrio)


# %%
# --------------------------------------------------
# 8. CONDICIONES INICIALES PARA EL DIAGRAMA DE FASE
# --------------------------------------------------

# Cada fila representa una condicion inicial distinta.
# Las columnas corresponden a [Y(0), T(0), z(0)].
#
# Incluimos las condiciones de la catedra y otras
# para analizar como cambia el comportamiento
# del sistema al modificar sus estados iniciales.

X0 = np.array([
    [1,   300, 300],    # Condicion original
    [0.1, 301, 301],    # Cercana al equilibrio
    [1,   350, 300],    # Mayor temperatura inicial
    [0.5, 300, 300],    # Menor combustible inicial
    [1,   300, 320]     # Mayor variable termica inicial
])

# Definimos un color para cada trayectoria.
colores = ['blue', 'red', 'green', 'orange', 'purple']



# --------------------------------------------------
# 9. DIAGRAMA DE FASE 3D - INTERVALO AMPLIADO
# --------------------------------------------------

# Extendemos el intervalo de simulacion a 100 segundos
# para observar mejor la evolucion de las trayectorias.
# La simulacion original de 2 s se conserva.

t_final_fase = 100
t_eval_fase = np.linspace(t_inicio, t_final_fase, N)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Recorremos las condiciones iniciales definidas en X0.
for i in range(len(X0)):

    x_inicial = X0[i, :]

    # Resolvemos el mismo modelo no lineal,
    # ahora durante 100 segundos.
    solucion = solve_ivp(
        modelo_nolineal,
        [t_inicio, t_final_fase],
        x_inicial,
        method='RK45',
        t_eval=t_eval_fase,
        rtol=1e-8,
        atol=1e-10
    )

    if not solucion.success:
        raise RuntimeError(solucion.message)

    Y = solucion.y[0, :]
    T = solucion.y[1, :]
    z = solucion.y[2, :]

    # Graficamos una trayectoria por cada condicion inicial.
    ax.plot(Y, T, z,
            color=colores[i],
            label=f'CI {i+1}: {X0[i, :].tolist()}')

    # Estrella: posicion inicial.
    ax.scatter(Y[0], T[0], z[0],
               color=colores[i], marker='*', s=120)

    # Circulo: posicion al finalizar la simulacion.
    ax.scatter(Y[-1], T[-1], z[-1],
               color=colores[i], marker='o', s=50)

# Marcamos el punto de equilibrio.
ax.scatter(xe[0], xe[1], xe[2],
           color='black', marker='X', s=130,
           label='Punto de equilibrio')

ax.set_xlabel('Y(t)')
ax.set_ylabel('T(t) [K]')
ax.set_zlabel('z(t) [K]')

ax.set_title('Modelo no lineal - Diagrama de fase (0 a 100 s)')

ax.view_init(elev=25, azim=-60)
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()



# %%
# --------------------------------------------------
# 10. CALCULO DE LA MATRIZ JACOBIANA
# --------------------------------------------------

# El jacobiano contiene las derivadas parciales
# de cada ecuacion respecto de cada estado.
#
# Se utiliza para aproximar el sistema no lineal
# alrededor de un punto de equilibrio.

# Agrupamos las ecuaciones en un vector.
f = sp.Matrix([f1, f2, f3])

# Agrupamos las variables de estado.
variables = sp.Matrix([Y_s, T_s, z_s])

# Calculamos el jacobiano general.
J = f.jacobian(variables)

print("\nMatriz jacobiana general:")
sp.pprint(J)

# Evaluamos el jacobiano en el equilibrio obtenido.
A_simbolica = J.subs({
    Y_s: xe[0],
    T_s: xe[1],
    z_s: xe[2]
})

print("\nMatriz A evaluada en el equilibrio:")
sp.pprint(A_simbolica)

# Convertimos la matriz a valores numericos.
A = np.array(A_simbolica, dtype=float)

print("\nMatriz A numerica:")
print(A)


# --------------------------------------------------
# 11. ANALISIS DE ESTABILIDAD
# --------------------------------------------------

# Calculamos los autovalores de la matriz A.
# Para un sistema continuo, si todos tienen
# parte real negativa, el equilibrio es
# localmente asintoticamente estable.

autovalores = np.linalg.eigvals(A)

print("\nAutovalores de la matriz A:")
print(autovalores)

# Verificamos el criterio de estabilidad.
if np.all(np.real(autovalores) < 0):

    print("\nEl punto de equilibrio es "
          "localmente asintoticamente estable.")

elif np.any(np.real(autovalores) > 0):

    print("\nEl punto de equilibrio es inestable.")

else:

    print("\nEl criterio de linealizacion "
          "no permite decidir la estabilidad.")

