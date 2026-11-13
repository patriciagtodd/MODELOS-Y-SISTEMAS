# -*- coding: utf-8 -*-
"""
Trabajo Final - Modelos y Sistemas | Modelo reducido de combustión
==================================================================

Estados:   x = (Y, T, z)
    Y  [-]  fracción de reactivo sin quemar (1 = todo sin quemar, 0 = todo consumido)
    T  [K]  temperatura de la zona de reacción
    z  [K]  variable auxiliar de temperatura (sigue a T y es forzada por Y)

Modelo:
    dY/dt = -k(T) * Y
    dT/dt =  q * k(T) * Y - h * (T - Ta)
    dz/dt =  alfa * (T - z) + beta * Y
    con  k(T) = k0 * exp(-ER / T)      (ley de Arrhenius)

Orden del archivo :
    0. Configuración
    1. Parámetros, condiciones iniciales y horizontes
    2. Modelo no lineal
    3. Equilibrio, Jacobiano y estabilidad
    4. Modelo linealizado 
    5. Simulaciones: diagramas de fase 3D y soluciones temporales (200 s y 1000 s)
    6. Error entre modelos
    7. Horizonte largo (8000 s)
    8. Condición inicial lejana del equilibrio
    9. Entrada térmica, función de transferencia, escalón e impulso
   10. Resumen de números para el informe

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy import signal

# %% 0. CONFIGURACIÓN
# Si GUARDAR = True, además de mostrar cada figura la guarda en la carpeta 'figuras'.
# (Las que usamos en el informe son las que se guardan con dpi=200.)
GUARDAR = True
CARPETA = 'figuras'
if GUARDAR:
    os.makedirs(CARPETA, exist_ok=True)


def mostrar(fig, nombre):
    """Guarda (si corresponde) y muestra la figura."""
    if GUARDAR:
        fig.savefig(os.path.join(CARPETA, nombre + '.png'), dpi=200, bbox_inches='tight')
    plt.show()


# %% 1. PARÁMETROS, CONDICIONES INICIALES Y HORIZONTES

k0, ER, q = 10**4, 5000, 100          # k0 [1/s], ER = E/R [K], q [K]
h, Ta, alfa, beta = 0.5, 300, 0.1, 10  # h [1/s], Ta [K], alfa [1/s], beta [K/s]
N = 5000                               # puntos de salida de cada simulación



             # [Y(0),T(0),z(0)].
X0 = np.array([[1,   300, 300],
               [0.1, 301, 301],
               [1,   350, 300],
               [0.5, 300, 300],
               [1,   300, 400]], dtype=float)

colores = ['blue', 'red', 'green', 'orange', 'purple'] 
etiquetas = [f'CI {i+1}: ({x[0]:g}, {x[1]:g}, {x[2]:g})' for i, x in enumerate(X0)]
nombres_estado = ['Y', 'T [K]', 'z [K]']

# Horizontes pedidos por la cátedra (el "2" original era un error de tipeo).
HORIZONTES = [200, 1000] #probamos tambien mil  para ver mejor el commportamiento

# OJO: la CI 5 tiene z(0) = 400 justo porque 300 + (beta/alfa)*Y(0) = 300 + 100*1 = 400,
# o sea que arranca "ya apoyada" en la curva a la que z tiende (ver sección 5).


def k_arrhenius(T):
    #Constante de reacción k(T) = k0 * exp(-ER/T)
    return k0 * np.exp(-ER / T)


# %% 2. MODELO NO LINEAL

def modelo_nolineal(t, x):
    Y, T, z = x
    reaccion = k_arrhenius(T) * Y            # velocidad de consumo del reactivo
    return [-reaccion,                       # dY/dt
            q * reaccion - h * (T - Ta),     # dT/dt: calor de la reacción - enfriamiento
            alfa * (T - z) + beta * Y]       # dz/dt


# %% 3. EQUILIBRIO, JACOBIANO Y ESTABILIDAD
# Lo hacemos simbólico con SymPy para no tener redondeos y para poder mostrar la cuenta.

Y_s, T_s, z_s = sp.symbols('Y T z', real=True)
variables = [Y_s, T_s, z_s]
# nsimplify convierte 0.5 -> 1/2 y 0.1 -> 1/10 (racionales exactos).
h_s, alfa_s = sp.nsimplify(h), sp.nsimplify(alfa)
kT_s = k0 * sp.exp(-ER / T_s)
f = sp.Matrix([-kT_s * Y_s,
               q * kT_s * Y_s - h_s * (T_s - Ta),
               alfa_s * (T_s - z_s) + beta * Y_s])

# A mano: k(T) > 0 siempre => dY/dt = 0 => Y = 0; después dT/dt = 0 da T = Ta
# y dz/dt = 0 da z = T. Es decir, xe = (0, 300, 300). SymPy lo confirma:
    
S = sp.solve([sp.Eq(fi, 0) for fi in f], variables, dict=True)
if not S:
    raise ValueError('No se encontraron puntos de equilibrio.')
xe_sym = [S[0][v] for v in variables]
xe = np.array([float(v) for v in xe_sym])

# Jacobiano general y evaluado en el equilibrio. Con Y = 0 se anulan los términos que
# tienen a Y multiplicando (por eso la nolinealidad "desaparece" al linealizar).
J = f.jacobian(variables)
A_s = J.subs(dict(zip(variables, xe_sym)))
A = np.array(A_s, dtype=float)
autovalores = np.linalg.eigvals(A)

a = k_arrhenius(Ta)   # a = k(Ta) = 5.78e-4 1/s  (reacción a temperatura ambiente)
tau = 1 / np.abs(np.sort(np.real(autovalores)))[::-1]   # constantes de tiempo, de la lenta a la rápida

print('=== 3. EQUILIBRIO Y ESTABILIDAD ===')
print('Puntos de equilibrio (Y, T, z):', [tuple(float(sol[v]) for v in variables) for sol in S])
print('Derivadas en el equilibrio (deben ser 0):', modelo_nolineal(0, xe))
print('Jacobiano evaluado A:\n', A)
print('Autovalores:', np.real(autovalores))
print('Estabilidad local asintótica (todos Re < 0):', bool(np.all(np.real(autovalores) < 0)))
print(f'a = k(Ta) = {a:.4e} 1/s')
print('Constantes de tiempo [s]:', np.round(tau, 2), '(la de la combustión es mucho más lenta)')

# ACLARACIÓN: como A es triangular inferior, los autovalores son los de la diagonal 
#(porque tambien es cuadrada): -a, -h y -alfa. Por eso salen reales y negativos sin necesidad de ninguna cuenta rara.
# Por el método indirecto de Lyapunov, xe es localmente asintóticamente estable.


# %% 4. MODELO LINEALIZADO (EN DESVIACIONES)
# dx = x - xe  =>  d(dx)/dt = A dx
def modelo_linealizado(t, dx):
    return A @ dx


# %% 5. SIMULACIONES
def simular(x0, t_final, n=N, metodo='RK45', rtol=1e-8, atol=1e-10):
    """Integra el modelo no lineal y el linealizado con la MISMA condición inicial.

    El linealizado trabaja con dx = x - xe, así que se le pasa x0 - xe y después se
    recupera el estado original sumando xe.
    """
    t = np.linspace(0, t_final, n)
    sol_nl = solve_ivp(modelo_nolineal, [0, t_final], x0, t_eval=t,
                       method=metodo, rtol=rtol, atol=atol)
    sol_lin = solve_ivp(modelo_linealizado, [0, t_final], np.asarray(x0) - xe, t_eval=t,
                        method=metodo, rtol=rtol, atol=atol)
    if not sol_nl.success or not sol_lin.success:
        raise RuntimeError('Falló una integración con solve_ivp.')
    return t, sol_nl.y, sol_lin.y + xe[:, None]


def figura_fases(sols_nl, sols_lin, t_final, nombre):
    """Diagramas de fase 3D (equivalente a plot3 de MATLAB), nolineal vs linealizado."""
    fig = plt.figure(figsize=(14, 6))
    for j, (titulo, sols) in enumerate([('No lineal', sols_nl), ('Linealizado', sols_lin)]):
        ax = fig.add_subplot(1, 2, j + 1, projection='3d')
        for i, x in enumerate(sols):
            ax.plot(x[0], x[1], x[2], color=colores[i], lw=1.6, label=etiquetas[i])
            ax.scatter(*x[:, 0], color=colores[i], marker='*', s=65)    # estrella = punto inicial
            ax.scatter(*x[:, -1], color=colores[i], marker='o', s=20)   # círculo = punto final
        ax.scatter(*xe, color='black', marker='X', s=95, label='Equilibrio')
        ax.set(xlabel='Y', ylabel='T [K]', zlabel='z [K]', title=titulo)
        ax.view_init(elev=22, azim=-60)
        if j == 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize=8)
    fig.suptitle(f'Diagramas de fase 3D - 0 a {t_final} s')
    fig.subplots_adjust(left=0.01, right=0.79, wspace=0.04)
    mostrar(fig, nombre)


def figura_temporal(t, sols_nl, sols_lin, t_final, nombre):
    """Soluciones temporales de los 3 estados, nolineal (izq.) vs linealizado (der.)."""
    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for j, (titulo, sols) in enumerate([('No lineal', sols_nl), ('Linealizado', sols_lin)]):
        for i, x in enumerate(sols):
            for k in range(3):
                ax[k, j].plot(t, x[k], color=colores[i], lw=1.4, label=etiquetas[i])
        ax[0, j].set_title(titulo)
        ax[2, j].set_xlabel('Tiempo [s]')
    for k, estado in enumerate(nombres_estado):
        ax[k, 0].set_ylabel(estado)
        for j in range(2):
            ax[k, j].grid(True, alpha=0.3)
    fig.suptitle(f'Soluciones temporales - 0 a {t_final} s')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    mostrar(fig, nombre)


print('\n=== 5. SIMULACIONES ===')
for t_final in HORIZONTES:
    sols_nl, sols_lin = [], []
    for x0 in X0:
        t, y_nl, y_lin = simular(x0, t_final)
        sols_nl.append(y_nl)
        sols_lin.append(y_lin)

    figura_fases(sols_nl, sols_lin, t_final, f'fases_{t_final}s')
    figura_temporal(t, sols_nl, sols_lin, t_final, f'temporal_{t_final}s')

# anotacion de lo que vemos:
# * Nolineal y linealizado se ven casi iguales. NO es un error del código: el equilibrio
#   tiene Y = 0 y la única nolinealidad es el producto k(T)*Y. Al linealizar, k(T) se
#   "congela" en k(Ta) = a, y el error que se comete es -(k(T) - a)*Y: chico si T no se
#   aleja mucho de Ta o si Y es chica.
# * Además T vuelve a Ta con constante de tiempo 2 s, así que los apartamientos de T
#   duran muy poco.
# * Y decae lento (tau ~ 1731 s). T queda cuasi-estacionaria en Ta + q*a*Y/h ~ 300.1 K.
#   z tiende a T + (beta/alfa)*Y ~ 300 + 100*Y, por eso sube hasta ~397 K y después baja
#   al ritmo de Y. La CI 5 (z(0)=400) ya arranca sobre esa curva.


# %% 6. ERROR ENTRE MODELOS (escala logarítmica)
# Para ver la diferencia que a simple vista no se nota. Usamos LSODA con tolerancias más
# finas (rtol=1e-9) para que el error numérico no tape la diferencia entre los modelos.

T_ERR = 1000
errores = []
print('\n=== 6. ERROR |nolineal - linealizado| (máximos en 0-%d s) ===' % T_ERR)
for i, x0 in enumerate(X0):
    t, y_nl, y_lin = simular(x0, T_ERR, metodo='LSODA', rtol=1e-9, atol=1e-11)
    e = np.abs(y_nl - y_lin)
    errores.append(e)
    print(f'CI {i+1}: Y {e[0].max():.2e} | T {e[1].max():.2e} K | z {e[2].max():.2e} K')

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
for k in range(3):
    for i, e in enumerate(errores):
        ax[k].semilogy(t, e[k] + 1e-16, color=colores[i], lw=1.3, label=f'CI {i+1}')
    ax[k].set_ylabel(f'|Δ{nombres_estado[k]}|')
    ax[k].set_ylim(1e-10, 1e2)
    ax[k].grid(True, alpha=0.3)
ax[2].set_xlabel('Tiempo [s]')
ax[0].legend(ncol=5, fontsize=8, loc='upper right')
fig.tight_layout()
mostrar(fig, 'error_modelos')

# la CI 1 y la CI 5 tienen exactamente el mismo error (la curva de la CI 1 queda
# tapada por la de la CI 5). Es lógico: z no realimenta ni a Y ni a T, y la ecuación de z
# es lineal (igual en los dos modelos), así que cambiar z(0) no cambia ningún error.
# La CI 3 (T(0) = 350 K) es la única con error apreciable (k(350) ~ 10.8 * k(300)).


# %% 7. HORIZONTE LARGO (8000 s)
# Los 1000 s de arriba son solo ~0.6 veces la constante de tiempo lenta (1731 s), así que ahí
# todavía no se ve llegar al equilibrio. Con 8000 s (~4.6 tau) sí.
T_LARGO = 8000
fig, ax = plt.subplots(3, 2, sharex=True, figsize=(9, 5.8))
print('\n=== 7. A t = %d s ===' % T_LARGO)
for i in (0, 2, 3):                       # CI 1, 3 y 4 (las otras no agregan nada)
    t, y_nl, y_lin = simular(X0[i], T_LARGO, n=8000, metodo='LSODA', rtol=1e-9, atol=1e-11)
    for j, y in enumerate((y_nl, y_lin)):
        for k in range(3):
            ax[k, j].plot(t, y[k], color=colores[i], lw=1.3, label=f'CI {i+1}')
    print(f'CI {i+1}: NL {np.round(y_nl[:, -1], 5)} | lineal {np.round(y_lin[:, -1], 5)}')
ax[0, 0].set_title('No lineal')
ax[0, 1].set_title('Linealizado')
for j in range(2):
    ax[2, j].set_xlabel('Tiempo [s]')
for k in range(3):
    ax[k, 0].set_ylabel(nombres_estado[k])
    for j in range(2):
        ax[k, j].grid(True, alpha=0.3)
ax[0, 0].legend(fontsize=8)
fig.tight_layout()
mostrar(fig, 'horizonte_largo')

# ACLARACIÓN: a 8000 s Y ~ 0.0098 y z ~ 301 K (todavía sigue a 300 + 100*Y). Ambos modelos
# convergen a xe, como predicen los autovalores negativos.


# %% 8. CONDICIÓN INICIAL LEJANA DEL EQUILIBRIO

# buscamos forzar que el linealizado falle: partimos de T(0) = 600 K, donde k(600) ~ 2.4 1/s
# (unas 4000 veces k(300)). La dinámica es rígida, por eso usamos Radau en vez de RK45.

print('\n=== 8. CONDICIÓN INICIAL LEJANA ===')
for T0 in (500, 600):
    t, y_nl, y_lin = simular([1, T0, 300], 20, n=2001, metodo='Radau', rtol=1e-9, atol=1e-11)
    print(f'T(0) = {T0} K: nolineal Y(20) = {y_nl[0, -1]:.3f} | lineal Y(20) = {y_lin[0, -1]:.3f}'
          f' | T máx nolineal = {y_nl[1].max():.1f} K')

t, y_nl, y_lin = simular([1, 600, 300], 20, n=20001, metodo='Radau', rtol=1e-9, atol=1e-11)
i_pico = y_nl[1].argmax()
print(f'T(0)=600: pico de T nolineal = {y_nl[1, i_pico]:.1f} K a t = {t[i_pico]:.2f} s | '
      f'Y < 0.1 a t = {t[np.argmax(y_nl[0] < 0.1)]:.2f} s | z(20): NL {y_nl[2, -1]:.1f} K vs lineal {y_lin[2, -1]:.1f} K')

# Mismos colores y estilo que las otras figuras: azul = nolineal, rojo punteado = linealizado.
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 4.8))
for k in range(3):
    ax[k].plot(t, y_nl[k], color='blue', lw=1.4, label='No lineal')
    ax[k].plot(t, y_lin[k], color='red', lw=1.4, ls='--', label='Linealizado')
    ax[k].set_ylabel(nombres_estado[k])
    ax[k].grid(True, alpha=0.3)
ax[0].legend(fontsize=8)
ax[2].set_xlabel('Tiempo [s]')
fig.suptitle('Condición inicial lejana (1, 600, 300) - 0 a 20 s')
fig.tight_layout(rect=[0, 0, 1, 0.96])
mostrar(fig, 'ci_lejana')



# %% 9. ENTRADA TÉRMICA Y FUNCIÓN DE TRANSFERENCIA
# El sistema es autónomo, así que le agregamos una entrada u(t) [K/s] en la ecuación de T
# (un aporte externo de calor, p. ej. una fuente térmica). Salida: y = deltaT = T - Ta.

B = np.array([[0], [1], [0]])
C = np.array([[0, 1, 0]])
D = np.array([[0]])
sistema = signal.StateSpace(A, B, C, D)     # representación en estados (A, B, C, D)

# H(s) = C (sI - A)^-1 B + D, calculada simbólicamente.
s_sym = sp.symbols('s')
H_s = sp.cancel((sp.Matrix(C) * (s_sym * sp.eye(3) - A_s).inv() * sp.Matrix(B) + sp.Matrix(D))[0])
print('\n=== 9. FUNCIÓN DE TRANSFERENCIA ===')
sp.pprint(H_s)

# Pasamos los coeficientes (sin escribirlos a mano) a scipy.signal.
num_s, den_s = sp.fraction(H_s)
num = [float(c) for c in sp.Poly(num_s, s_sym).all_coeffs()]
den = [float(c) for c in sp.Poly(den_s, s_sym).all_coeffs()]
H = signal.TransferFunction(num, den)
print('Polos de H:', H.poles, '| ganancia estática H(0) =', num[-1] / den[-1])

# ACLARACIÓN: H(s) = 1/(s+h) = 2/(2s+1) es de orden 1 aunque el sistema es de orden 3.
# Los polos -a y -alfa no aparecen: Y no recibe la entrada (no controlable) y z no
# realimenta a T (no observable desde la salida deltaT).

# Respuestas al escalón y al impulso (condiciones iniciales nulas en las desviaciones).
t_resp = np.linspace(0, 20, 1000)
t_esc, y_esc = signal.step(H, T=t_resp)
t_imp, y_imp = signal.impulse(H, T=t_resp)

# Chequeo contra las fórmulas analíticas: escalón = (1/h)(1 - e^{-h t}), impulso = e^{-h t}
print('máx |escalón numérico - analítico| =', np.abs(y_esc - (1 - np.exp(-h * t_esc)) / h).max())
print('máx |impulso numérico - analítico| =', np.abs(y_imp - np.exp(-h * t_imp)).max())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t_esc, y_esc)
ax[0].set_title('Respuesta al escalón unitario')
ax[1].plot(t_imp, y_imp)
ax[1].set_title('Respuesta al impulso unitario')
for ejes in ax:
    ejes.set_xlabel('Tiempo [s]')
    ejes.set_ylabel('Variación de temperatura')
    ejes.grid(True, alpha=0.3)
fig.tight_layout()
mostrar(fig, 'escalon_impulso')


# %% 10. RESUMEN DE NÚMEROS 

print('\n=== 10. RESUMEN ===')
print('xe =', xe)
print('autovalores =', np.round(np.sort(np.real(autovalores)), 6))
print('tau [s] =', np.round(tau, 1))
print('H(s) = ', H_s)
print('k(350)/k(300) =', round(k_arrhenius(350) / k_arrhenius(300), 2))
print('k(600)/k(300) =', round(k_arrhenius(600) / k_arrhenius(300), 0))
