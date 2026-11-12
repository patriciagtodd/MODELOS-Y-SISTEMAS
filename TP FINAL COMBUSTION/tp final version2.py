# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:26:33 2026

@author: Aragon
"""

"""
Trabajo Final - Modelos y Sistemas
Modelo reducido de combustion

Parte 1: Modelo no lineal
Parte 2: Puntos de equilibrio
Parte 3: Simulaciones y diagramas de fase
Parte 4: Jacobiano y estabilidad
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
# 2. CONDICIONES INICIALES DE LA CATEDRA
# --------------------------------------------------

Y0 = 1           # Concentracion inicial de combustible
T0 = 300         # Temperatura inicial [K]
z0 = 300         # Variable termica interna inicial [K]

# Agrupamos los tres estados iniciales en un vector.
x0 = [Y0, T0, z0]


# --------------------------------------------------
# 3. INTERVALO DE SIMULACION
# --------------------------------------------------

t_inicio = 0
t_final = 2
N = 5000

# Creamos 5000 instantes entre 0 y 2 segundos.
# Al incluir ambos extremos tenemos N-1 intervalos.

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
# 5. CALCULO DE LOS PUNTOS DE EQUILIBRIO
# --------------------------------------------------

# Seguimos el procedimiento utilizado por la profesora:
#
# 1. Definir las variables simbolicas.
# 2. Igualar las ecuaciones diferenciales a cero.
# 3. Resolver simultaneamente mediante solve.
# 4. Convertir los resultados a valores numericos.

Y_s, T_s, z_s = sp.symbols('Y T z', real=True)

# Definimos las ecuaciones simbolicas.

f1 = -k0 * sp.exp(-ER/T_s) * Y_s

f2 = q * k0 * sp.exp(-ER/T_s) * Y_s - h * (T_s - Ta)

f3 = alfa * (T_s - z_s) + beta * Y_s

# Igualamos las tres derivadas a cero.

eqs = [
    sp.Eq(f1, 0),
    sp.Eq(f2, 0),
    sp.Eq(f3, 0)
]

# Resolvemos el sistema de ecuaciones.

S = sp.solve(eqs, [Y_s, T_s, z_s], dict=True)

print("\nPuntos de equilibrio simbolicos:")
print(S)

# Convertimos las soluciones a valores numericos.
# Cada fila de Xcrit representa un punto de equilibrio.

Xcrit = np.array([
    [float(s[Y_s]), float(s[T_s]), float(s[z_s])]
    for s in S
])

print("\nPuntos de equilibrio numericos:")
print(Xcrit)

# Seleccionamos el punto de equilibrio encontrado.

xe = Xcrit[0, :]

# Comprobamos que las derivadas sean cero.

derivadas_equilibrio = modelo_nolineal(0, xe)

print("\nDerivadas en el equilibrio:")
print(derivadas_equilibrio)


# %%
# --------------------------------------------------
# 6. CONDICIONES INICIALES PARA LAS SIMULACIONES
# --------------------------------------------------

# Cada fila representa una condicion inicial.
# Cada columna corresponde a [Y(0), T(0), z(0)].
#
# Elegimos diferentes condiciones iniciales para
# estudiar como varia la dinamica del sistema.
#
# Incluimos la condicion original proporcionada
# por la catedra.

X0 = np.array([
    [1,   300, 300],    # CI 1: condicion original
    [0.1, 301, 301],    # CI 2: cercana al equilibrio
    [1,   350, 300],    # CI 3: mayor temperatura
    [0.5, 300, 300],    # CI 4: menor combustible
    [1,   300, 320]     # CI 5: mayor variable interna
])

# Asignamos un color a cada condicion inicial.
# Mantendremos los mismos colores en todos los
# graficos para facilitar las comparaciones.

colores = ['blue', 'red', 'green', 'orange', 'purple']

# Creamos etiquetas claras para las leyendas.
# Evitamos mostrar los vectores numpy directamente,
# porque pueden aparecer en notacion cientifica.

etiquetas = [
    'CI 1: (1, 300, 300)',
    'CI 2: (0.1, 301, 301)',
    'CI 3: (1, 350, 300)',
    'CI 4: (0.5, 300, 300)',
    'CI 5: (1, 300, 320)'
]


# %%
# --------------------------------------------------
# 7. SIMULACIONES DEL MODELO NO LINEAL - 2 SEGUNDOS
# --------------------------------------------------

# Primero utilizamos el intervalo de simulacion
# proporcionado por la catedra: t entre 0 y 2 s.
#
# Resolvemos el sistema para cada condicion inicial.
# Guardamos los resultados para utilizarlos tanto
# en las soluciones temporales como en el diagrama 3D.

soluciones_2s = []

for i in range(len(X0)):

    x_inicial = X0[i, :]

    sol = solve_ivp(
        modelo_nolineal,
        [t_inicio, t_final],
        x_inicial,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    # Verificamos que la integracion haya sido exitosa.
    if not sol.success:
        raise RuntimeError(sol.message)

    # Guardamos la solucion correspondiente.
    soluciones_2s.append(sol)

print("\nSimulaciones de 2 segundos completadas.")


# --------------------------------------------------
# 8. SOLUCIONES TEMPORALES - 2 SEGUNDOS
# --------------------------------------------------

# Graficamos las tres variables de estado.
#
# Cada grafico contiene las cinco condiciones
# iniciales diferenciadas por colores.
#
# De esta manera podemos comparar como evoluciona
# cada variable al modificar su valor inicial.

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    sol = soluciones_2s[i]

    tt = sol.t
    YY = sol.y[0, :]
    TT = sol.y[1, :]
    zz = sol.y[2, :]

    # Concentracion de combustible.
    ax[0].plot(tt, YY,
               color=colores[i],
               label=etiquetas[i])

    # Temperatura.
    ax[1].plot(tt, TT,
               color=colores[i])

    # Variable termica interna.
    ax[2].plot(tt, zz,
               color=colores[i])

# Identificamos los ejes y los graficos.

ax[0].set_ylabel('Y(t)')
ax[0].set_title('Soluciones temporales - 0 a 2 s')

ax[1].set_ylabel('T(t) [K]')

ax[2].set_ylabel('z(t) [K]')
ax[2].set_xlabel('Tiempo [s]')

# Agregamos cuadriculas para facilitar la lectura.

for a in ax:
    a.grid(True, alpha=0.4)

# Ubicamos una sola leyenda para toda la figura.
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3, fontsize=9)

# Dejamos espacio para la leyenda.
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 9. DIAGRAMA DE FASE 3D - 2 SEGUNDOS
# --------------------------------------------------

# Representamos las trayectorias en el espacio
# de estados utilizando las mismas simulaciones.
#
# Eje X: concentracion de combustible Y
# Eje Y: temperatura T
# Eje Z: variable termica interna z
#
# Las estrellas indican condiciones iniciales.
# Los circulos indican puntos finales.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X0)):

    sol = soluciones_2s[i]

    Y = sol.y[0, :]
    T = sol.y[1, :]
    z = sol.y[2, :]

    # Trayectoria de cada condicion inicial.
    ax.plot(Y, T, z,
            color=colores[i],
            linewidth=1.8,
            label=etiquetas[i])

    # Punto inicial: estrella.
    ax.scatter(Y[0], T[0], z[0],
               color=colores[i],
               marker='*',
               s=75,
               depthshade=False)

    # Punto final: circulo.
    ax.scatter(Y[-1], T[-1], z[-1],
               color=colores[i],
               marker='o',
               s=25,
               depthshade=False)

# Punto de equilibrio.
ax.scatter(xe[0], xe[1], xe[2],
           color='black',
           marker='X',
           s=110,
           label='Equilibrio',
           depthshade=False)

ax.set_xlabel('Y(t)', labelpad=9)
ax.set_ylabel('T(t) [K]', labelpad=9)
ax.set_zlabel('z(t) [K]', labelpad=9)

ax.set_title('Diagrama de fase 3D - 0 a 2 s')

# Ajustamos el angulo de observacion.
ax.view_init(elev=22, azim=-60)

# Colocamos la leyenda a la derecha para evitar
# que tape las trayectorias.
ax.legend(loc='upper left',
          bbox_to_anchor=(1.02, 1),
          fontsize=8)

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------
# 10. SIMULACIONES DEL MODELO NO LINEAL - 100 SEGUNDOS
# --------------------------------------------------

# Extendemos el intervalo a 100 segundos para
# estudiar la evolucion del sistema en un tiempo mayor.
#
# Este experimento es complementario.
# No reemplaza la simulacion original de 2 s.

t_final_fase = 100
t_eval_fase = np.linspace(t_inicio, t_final_fase, N)

# Guardamos las cinco soluciones del nuevo intervalo.

soluciones_100s = []

for i in range(len(X0)):

    x_inicial = X0[i, :]

    sol = solve_ivp(
        modelo_nolineal,
        [t_inicio, t_final_fase],
        x_inicial,
        method='RK45',
        t_eval=t_eval_fase,
        rtol=1e-8,
        atol=1e-10
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    soluciones_100s.append(sol)

print("\nSimulaciones de 100 segundos completadas.")


# --------------------------------------------------
# 11. SOLUCIONES TEMPORALES - 100 SEGUNDOS
# --------------------------------------------------

# Repetimos los graficos temporales para las mismas
# cinco condiciones iniciales, pero durante 100 s.
#
# Esto permite observar cambios que no son evidentes
# durante los primeros dos segundos.

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    sol = soluciones_100s[i]

    tt = sol.t
    YY = sol.y[0, :]
    TT = sol.y[1, :]
    zz = sol.y[2, :]

    ax[0].plot(tt, YY,
               color=colores[i],
               label=etiquetas[i])

    ax[1].plot(tt, TT,
               color=colores[i])

    ax[2].plot(tt, zz,
               color=colores[i])

ax[0].set_ylabel('Y(t)')
ax[0].set_title('Soluciones temporales - 0 a 100 s')

ax[1].set_ylabel('T(t) [K]')

ax[2].set_ylabel('z(t) [K]')
ax[2].set_xlabel('Tiempo [s]')

for a in ax:
    a.grid(True, alpha=0.4)

fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3, fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 12. DIAGRAMA DE FASE 3D - 100 SEGUNDOS
# --------------------------------------------------

# Construimos el segundo diagrama de fase
# utilizando las simulaciones del intervalo ampliado.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X0)):

    sol = soluciones_100s[i]

    Y = sol.y[0, :]
    T = sol.y[1, :]
    z = sol.y[2, :]

    # Trayectoria.
    ax.plot(Y, T, z,
            color=colores[i],
            linewidth=1.8,
            label=etiquetas[i])

    # Condicion inicial.
    ax.scatter(Y[0], T[0], z[0],
               color=colores[i],
               marker='*',
               s=75,
               depthshade=False)

    # Punto final.
    ax.scatter(Y[-1], T[-1], z[-1],
               color=colores[i],
               marker='o',
               s=25,
               depthshade=False)

# Punto de equilibrio.
ax.scatter(xe[0], xe[1], xe[2],
           color='black',
           marker='X',
           s=110,
           label='Equilibrio',
           depthshade=False)

ax.set_xlabel('Y(t)', labelpad=9)
ax.set_ylabel('T(t) [K]', labelpad=9)
ax.set_zlabel('z(t) [K]', labelpad=9)

ax.set_title('Diagrama de fase 3D - 0 a 100 s')

ax.view_init(elev=22, azim=-60)

ax.legend(loc='upper left',
          bbox_to_anchor=(1.02, 1),
          fontsize=8)

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------
# 13. CALCULO DE LA MATRIZ JACOBIANA
# --------------------------------------------------

# El jacobiano contiene las derivadas parciales
# de cada ecuacion respecto de cada variable de estado.
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
# 14. ANALISIS DE ESTABILIDAD
# --------------------------------------------------

# Calculamos los autovalores de la matriz A.
#
# Si todos tienen parte real negativa,
# el equilibrio es localmente asintoticamente estable.

autovalores = np.linalg.eigvals(A)

print("\nAutovalores de la matriz A:")
print(autovalores)

if np.all(np.real(autovalores) < 0):

    print("\nEl punto de equilibrio es "
          "localmente asintoticamente estable.")

elif np.any(np.real(autovalores) > 0):

    print("\nEl punto de equilibrio es inestable.")

else:

    print("\nEl criterio de linealizacion "
          "no permite decidir la estabilidad.")

# %%
# --------------------------------------------------
# 15. DEFINICION DEL MODELO LINEALIZADO
# --------------------------------------------------

# El modelo linealizado se obtiene utilizando
# la matriz A calculada a partir del jacobiano.
#
# Trabajamos con desviaciones respecto del equilibrio:
#
# dx = x - xe
#
# La ecuacion diferencial es:
#
# d(dx)/dt = A @ dx
#
# El simbolo @ representa el producto matricial
# en Python.

def modelo_linealizado(t, dx):

    # Calculamos las derivadas de las desviaciones
    # utilizando la matriz A.

    ddxdt = A @ dx

    # Devolvemos las tres derivadas.
    return ddxdt

# %%


# --------------------------------------------------
# 16. SIMULACIONES DEL MODELO LINEALIZADO - 2 SEGUNDOS
# --------------------------------------------------

# Guardamos las soluciones linealizadas en una lista,
# al igual que hicimos para el modelo no lineal.

soluciones_lin_2s = []

for i in range(len(X0)):

    # Seleccionamos la condicion inicial original.
    x_inicial = X0[i, :]

    # Como el modelo linealizado trabaja con
    # desviaciones respecto del equilibrio,
    # debemos restar el punto de equilibrio.

    dx0 = x_inicial - xe

    # Resolvemos las ecuaciones linealizadas.
    sol_lin = solve_ivp(
        modelo_linealizado,
        [t_inicio, t_final],
        dx0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )

    # Comprobamos que la simulacion sea correcta.
    if not sol_lin.success:
        raise RuntimeError(sol_lin.message)

    # Recuperamos las variables de estado originales.
    # Sumamos el equilibrio a cada columna
    # de la matriz de soluciones.

    x_lineal = sol_lin.y + xe[:, None]

    # Guardamos las soluciones fisicas.
    soluciones_lin_2s.append(x_lineal)

print("\nSimulaciones linealizadas de 2 segundos completadas.")

# %%

# --------------------------------------------------
# 17. COMPARACION TEMPORAL - CONDICION INICIAL ORIGINAL
# --------------------------------------------------

# Seleccionamos la primera condicion inicial,
# correspondiente a los valores de la catedra.

sol_nl = soluciones_2s[0]
x_lin = soluciones_lin_2s[0]

tt = sol_nl.t

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

# Concentracion de combustible.
ax[0].plot(tt, sol_nl.y[0, :],
           'b-', label='No lineal')

ax[0].plot(tt, x_lin[0, :],
           'r--', label='Linealizado')

ax[0].set_ylabel('Y(t)')
ax[0].set_title('Comparacion de modelos - CI original')
ax[0].legend()
ax[0].grid(True, alpha=0.4)

# Temperatura.
ax[1].plot(tt, sol_nl.y[1, :],
           'b-', label='No lineal')

ax[1].plot(tt, x_lin[1, :],
           'r--', label='Linealizado')

ax[1].set_ylabel('T(t) [K]')
ax[1].legend()
ax[1].grid(True, alpha=0.4)

# Variable termica interna.
ax[2].plot(tt, sol_nl.y[2, :],
           'b-', label='No lineal')

ax[2].plot(tt, x_lin[2, :],
           'r--', label='Linealizado')

ax[2].set_ylabel('z(t) [K]')
ax[2].set_xlabel('Tiempo [s]')
ax[2].legend()
ax[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

# %%

# --------------------------------------------------
# 18. DIAGRAMA DE FASE 3D - MODELO LINEALIZADO (2 s)
# --------------------------------------------------

# Graficamos las cinco condiciones iniciales
# del modelo linealizado en el espacio de estados.
#
# Utilizamos las mismas condiciones iniciales,
# colores e intervalo de tiempo que en el modelo
# no lineal, para permitir la comparacion.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X0)):

    # Recuperamos las soluciones fisicas del modelo
    # linealizado, previamente calculadas.
    x_lineal = soluciones_lin_2s[i]

    Y = x_lineal[0, :]
    T = x_lineal[1, :]
    z = x_lineal[2, :]

    # Graficamos cada trayectoria.
    ax.plot(Y, T, z,
            color=colores[i],
            linewidth=1.8,
            label=etiquetas[i])

    # Marcamos la condicion inicial con una estrella.
    ax.scatter(Y[0], T[0], z[0],
               color=colores[i],
               marker='*',
               s=75,
               depthshade=False)

    # Marcamos el punto final con un circulo.
    ax.scatter(Y[-1], T[-1], z[-1],
               color=colores[i],
               marker='o',
               s=25,
               depthshade=False)

# Marcamos el mismo punto de equilibrio.
ax.scatter(xe[0], xe[1], xe[2],
           color='black',
           marker='X',
           s=110,
           label='Punto de equilibrio',
           depthshade=False)

# Identificamos las variables de estado.
ax.set_xlabel('Y(t)', labelpad=9)
ax.set_ylabel('T(t) [K]', labelpad=9)
ax.set_zlabel('z(t) [K]', labelpad=9)

ax.set_title('Diagrama de fase 3D - Modelo linealizado (0 a 2 s)')

# Utilizamos el mismo angulo de observacion
# que en el grafico del modelo no lineal.
ax.view_init(elev=22, azim=-60)

# Colocamos la leyenda fuera del grafico.
ax.legend(loc='upper left',
          bbox_to_anchor=(1.02, 1),
          fontsize=8)

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------
# 19. SOLUCIONES TEMPORALES DEL MODELO LINEALIZADO
#     PARA DISTINTAS CONDICIONES INICIALES (2 s)
# --------------------------------------------------

# Graficamos la evolucion temporal de las tres
# variables de estado del modelo linealizado.
#
# Utilizamos las mismas cinco condiciones iniciales
# y los mismos colores que en el modelo no lineal.
#
# De esta manera podremos comparar los resultados
# de ambos modelos bajo las mismas condiciones.

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    # Recuperamos los estados fisicos del modelo
    # linealizado para la condicion inicial i.
    x_lineal = soluciones_lin_2s[i]

    # Separamos las tres variables de estado.
    Y_lin = x_lineal[0, :]
    T_lin = x_lineal[1, :]
    z_lin = x_lineal[2, :]

    # Concentracion de combustible.
    ax[0].plot(t_eval, Y_lin,
               color=colores[i],
               label=etiquetas[i])

    # Temperatura.
    ax[1].plot(t_eval, T_lin,
               color=colores[i])

    # Variable termica interna.
    ax[2].plot(t_eval, z_lin,
               color=colores[i])

# Identificamos los ejes y las variables.
ax[0].set_ylabel('Y(t)')
ax[0].set_title('Modelo linealizado - Soluciones temporales (0 a 2 s)')

ax[1].set_ylabel('T(t) [K]')

ax[2].set_ylabel('z(t) [K]')
ax[2].set_xlabel('Tiempo [s]')

# Agregamos cuadriculas.
for a in ax:
    a.grid(True, alpha=0.4)

# Una unica leyenda para las cinco condiciones.
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3,
           fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 20. SIMULACIONES DEL MODELO LINEALIZADO (100 s)
# --------------------------------------------------

# Repetimos las simulaciones durante 100 segundos.
#
# Utilizamos las mismas condiciones iniciales
# del modelo no lineal y el mismo intervalo
# temporal para poder comparar ambos resultados.

soluciones_lin_100s = []

for i in range(len(X0)):

    # Condicion inicial fisica.
    x_inicial = X0[i, :]

    # Calculamos la desviacion respecto del equilibrio.
    dx0 = x_inicial - xe

    # Resolvemos el modelo linealizado.
    sol_lin = solve_ivp(
        modelo_linealizado,
        [t_inicio, t_final_fase],
        dx0,
        method='RK45',
        t_eval=t_eval_fase,
        rtol=1e-8,
        atol=1e-10
    )

    # Verificamos el resultado.
    if not sol_lin.success:
        raise RuntimeError(sol_lin.message)

    # Recuperamos las variables fisicas.
    x_lineal = sol_lin.y + xe[:, None]

    # Guardamos las soluciones.
    soluciones_lin_100s.append(x_lineal)

print("\nSimulaciones linealizadas de 100 segundos completadas.")


# --------------------------------------------------
# 21. SOLUCIONES TEMPORALES DEL LINEALIZADO (100 s)
# --------------------------------------------------

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    x_lineal = soluciones_lin_100s[i]

    Y_lin = x_lineal[0, :]
    T_lin = x_lineal[1, :]
    z_lin = x_lineal[2, :]

    ax[0].plot(t_eval_fase, Y_lin,
               color=colores[i],
               label=etiquetas[i])

    ax[1].plot(t_eval_fase, T_lin,
               color=colores[i])

    ax[2].plot(t_eval_fase, z_lin,
               color=colores[i])

ax[0].set_ylabel('Y(t)')
ax[0].set_title('Modelo linealizado - Soluciones temporales (0 a 100 s)')

ax[1].set_ylabel('T(t) [K]')

ax[2].set_ylabel('z(t) [K]')
ax[2].set_xlabel('Tiempo [s]')

for a in ax:
    a.grid(True, alpha=0.4)

fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3,
           fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 22. DIAGRAMA DE FASE 3D - LINEALIZADO (100 s)
# --------------------------------------------------

# Representamos las cinco trayectorias del modelo
# linealizado durante un intervalo de 100 segundos.
#
# Utilizamos las mismas condiciones iniciales,
# colores, intervalo temporal y angulo de observacion
# que en el diagrama del modelo no lineal.
#
# Esto nos permitira comparar ambos modelos.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X0)):

    # Recuperamos las soluciones del modelo linealizado.
    x_lineal = soluciones_lin_100s[i]

    # Separamos las tres variables de estado.
    Y = x_lineal[0, :]
    T = x_lineal[1, :]
    z = x_lineal[2, :]

    # Graficamos cada trayectoria.
    ax.plot(Y, T, z,
            color=colores[i],
            linewidth=1.8,
            label=etiquetas[i])

    # Marcamos la condicion inicial con una estrella.
    ax.scatter(Y[0], T[0], z[0],
               color=colores[i],
               marker='*',
               s=75,
               depthshade=False)

    # Marcamos el punto final con un circulo.
    ax.scatter(Y[-1], T[-1], z[-1],
               color=colores[i],
               marker='o',
               s=25,
               depthshade=False)

# Marcamos el punto de equilibrio.
ax.scatter(xe[0], xe[1], xe[2],
           color='black',
           marker='X',
           s=110,
           label='Punto de equilibrio',
           depthshade=False)

# Identificamos los ejes.
ax.set_xlabel('Y(t)', labelpad=9)
ax.set_ylabel('T(t) [K]', labelpad=9)
ax.set_zlabel('z(t) [K]', labelpad=9)

ax.set_title('Diagrama de fase 3D - Linealizado (0 a 100 s)')

# Mantenemos el mismo angulo del modelo no lineal.
ax.view_init(elev=22, azim=-60)

# Colocamos la leyenda fuera de las trayectorias.
ax.legend(loc='upper left',
          bbox_to_anchor=(1.02, 1),
          fontsize=8)

plt.tight_layout()
plt.show()


# %%
# --------------------------------------------------
# 23. ERROR ENTRE MODELO NO LINEAL Y LINEALIZADO
#     INTERVALO DE 100 SEGUNDOS
# --------------------------------------------------

# Calculamos el error absoluto entre las soluciones
# del modelo no lineal y del modelo linealizado.
#
# Utilizamos las mismas condiciones iniciales
# y los mismos instantes de tiempo.
#
# El error absoluto permite comparar directamente
# las diferencias de cada variable de estado.

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    # Recuperamos la solucion del modelo no lineal.
    x_nolineal = soluciones_100s[i].y

    # Recuperamos la solucion del modelo linealizado.
    x_lineal = soluciones_lin_100s[i]

    # Calculamos el valor absoluto de la diferencia
    # entre las soluciones en cada instante.
    error = np.abs(x_nolineal - x_lineal)

    # Separamos los errores de las tres variables.
    error_Y = error[0, :]
    error_T = error[1, :]
    error_z = error[2, :]

    # Error de concentracion.
    ax[0].plot(t_eval_fase, error_Y,
               color=colores[i],
               label=etiquetas[i])

    # Error de temperatura.
    ax[1].plot(t_eval_fase, error_T,
               color=colores[i])

    # Error de la variable termica interna.
    ax[2].plot(t_eval_fase, error_z,
               color=colores[i])

# Identificamos cada grafico.
ax[0].set_ylabel('Error absoluto Y')
ax[0].set_title('Error entre modelos - 0 a 100 s')

ax[1].set_ylabel('Error absoluto T [K]')

ax[2].set_ylabel('Error absoluto z [K]')
ax[2].set_xlabel('Tiempo [s]')

# Agregamos cuadriculas para facilitar la lectura.
for a in ax:
    a.grid(True, alpha=0.4)

# Una leyenda para identificar las cinco condiciones.
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3,
           fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 24. ERROR ENTRE MODELO NO LINEAL Y LINEALIZADO
#     INTERVALO DE 2 SEGUNDOS
# --------------------------------------------------

# Calculamos el error absoluto entre ambos modelos
# para las mismas cinco condiciones iniciales.
#
# Utilizamos el intervalo original de la catedra:
# t entre 0 y 2 segundos.
#
# El error absoluto se calcula como:
# error = |modelo no lineal - modelo linealizado|

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

for i in range(len(X0)):

    # Recuperamos la solucion del modelo no lineal.
    x_nolineal = soluciones_2s[i].y

    # Recuperamos la solucion del modelo linealizado.
    x_lineal = soluciones_lin_2s[i]

    # Calculamos el error absoluto en cada instante.
    error = np.abs(x_nolineal - x_lineal)

    # Separamos el error de cada variable.
    error_Y = error[0, :]
    error_T = error[1, :]
    error_z = error[2, :]

    # Error de concentracion de combustible.
    ax[0].plot(t_eval, error_Y,
               color=colores[i],
               label=etiquetas[i])

    # Error de temperatura.
    ax[1].plot(t_eval, error_T,
               color=colores[i])

    # Error de la variable termica interna.
    ax[2].plot(t_eval, error_z,
               color=colores[i])

# Identificamos los graficos.
ax[0].set_ylabel('Error absoluto Y')
ax[0].set_title('Error entre modelos - 0 a 2 s')

ax[1].set_ylabel('Error absoluto T [K]')

ax[2].set_ylabel('Error absoluto z [K]')
ax[2].set_xlabel('Tiempo [s]')

# Agregamos cuadriculas.
for a in ax:
    a.grid(True, alpha=0.4)

# Una unica leyenda para las cinco condiciones.
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, 0.01),
           ncol=3,
           fontsize=9)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()


# %%
# --------------------------------------------------
# 26. MODELO LINEALIZADO CON ENTRADA EXTERNA
# --------------------------------------------------

# El modelo original es autonomo.
# Incorporamos una entrada termica externa u(t)
# que actua sobre la ecuacion de temperatura.
#
# La entrada representa una tasa de calentamiento
# o enfriamiento externo [K/s].
#
# Elegimos como salida la variacion de temperatura
# respecto del equilibrio: deltaT = T - Ta.

# Matriz de entrada B.
# La entrada afecta solamente a la segunda ecuacion.

B = np.array([
    [0],
    [1],
    [0]
])

# Matriz de salida C.
# Observamos la temperatura, segundo estado.

C = np.array([[0, 1, 0]])

# Matriz de transmision directa D.
# No existe efecto instantaneo de la entrada
# sobre la temperatura de salida.

D = np.array([[0]])

print("\nMatriz B:")
print(B)

print("\nMatriz C:")
print(C)

print("\nMatriz D:")
print(D)


# --------------------------------------------------
# 27. CALCULO DE LA FUNCION DE TRANSFERENCIA
# --------------------------------------------------

from scipy import signal

# Convertimos el sistema en espacio de estados
# a una funcion de transferencia.
#
# ss2tf devuelve los coeficientes del numerador
# y del denominador de H(s).

num, den = signal.ss2tf(A, B, C, D)

# Como tenemos una unica salida,
# seleccionamos la primera fila del numerador.

num = num[0]

print("\nNumerador de H(s):")
print(num)

print("\nDenominador de H(s):")
print(den)

# Construimos la funcion de transferencia.
H = signal.TransferFunction(num, den)

print("\nFuncion de transferencia:")
print(H)

# --------------------------------------------------
# 28. FUNCION DE TRANSFERENCIA SIMPLIFICADA
# --------------------------------------------------

# La funcion obtenida con ss2tf contiene factores
# comunes en el numerador y denominador.
#
# Para simplificarla utilizamos SymPy, que permite
# realizar la cancelacion algebraica.

s = sp.symbols('s')

# Utilizamos la matriz simbolica del sistema.
# Asi evitamos errores de redondeo al simplificar.

A_s = sp.Matrix(A_simbolica)

B_s = sp.Matrix(B)
C_s = sp.Matrix(C)
D_s = sp.Matrix(D)

# Calculamos H(s) a partir de las matrices.
H_s = (C_s * (s * sp.eye(3) - A_s).inv() * B_s + D_s)[0]

# Simplificamos la expresion simbolica.
H_simplificada = sp.cancel(H_s)

print("\nFuncion de transferencia simplificada:")
sp.pprint(H_simplificada)

# Definimos la funcion de transferencia simplificada
# para utilizarla en las siguientes simulaciones.

H = signal.TransferFunction([1], [1, 0.5])

print("\nFuncion de transferencia para las simulaciones:")
print(H)
