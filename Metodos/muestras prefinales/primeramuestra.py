'''
- ME DAN LA FUNCION Y LA MISMA IGUALADA A 1
- ME DICEN QUE DEFINA UNA FUNCIONCONVENIENTE PARA HALLAR LAS 
SOLUCIONES DE LA ECUACION Y GRAFICARLA A VER QUE METODO CONVIENE
'''
import numpy as np 
import matplotlib.pyplot as plt
from funciones import fcbiseccion

#MAIN 
'''
# Función y su derivada
f1 = lambda x: (np.exp(x**4 -4*x**3 + 2*x**2 + 8*x - 8) + x - 3)/(x - 2) - 1
inter = [0, 9/5]
tol = 1e-4 #esto es 0,001 y al pedir menor que esto ya son las 4

# Gráfico
x = np.linspace(0, 2, 100)
y1 = f1(x)
plt.plot(x, y1, 'r', label="f(x)")

# Ejes coordenados
plt.axhline(0, color='k', linestyle='--')
plt.axvline(0, color='k', linestyle='--')
plt.axvline(9/5, color='r', linestyle='--')


plt.legend()
plt.grid(True)
plt.show()

# EN EL INTERVALO [0,9/5] SOLO PARECE HABER UNS SOLUCION

# VEO QUE ONDA LA BISECCION
x0 = fcbiseccion(f1, inter, tol) '''

''' sabiendo que no necesite de muchas iteraciones para converger con gran excatitud
    me parece innecesario perder el tiempo en calcular una derivada que no me asegura la convergencia'''


#==========================================================================================================
'''#EJERCICIO 2
f2 = lambda x: np.sin(np.log(3*x))
inter = [1, 8]
tol = 1e-4

# Gráfico de la función
x = np.linspace(1, 10, 200)
y2 = f2(x)
plt.plot(x, y2, 'r', label="f(x)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

# Aplicar bisección
x01 = fcbiseccion(f2, inter, tol)

# Intervalo y número de nodos
a, b = 1, x01
n = 6   # 7 nodos → 6 subintervalos
x = np.linspace(a, b, n+1)
y = f2(x)

# uso trapecio compuesto con 6 nodos porque...
from funciones import fc_TrapreciosCompuesto
Itrap = fc_TrapreciosCompuesto(f2, a, b, h)
print(f"Área por trapecio compuesto (7 nodos): {Itrap:.6f}")

#uso simpson compuesto porque...
from funciones import fc_SimpsonCompuesto
Isimp = fc_SimpsonCompuesto(f2, a, b, h)
print(f"Área por Simpson 1/3 compuesto (7 nodos): {Isimp:.6f}")

# 4️⃣ Cota de error teórica

# Para trapecio: |E_T| <= ((b-a)^3 / (12n^2)) * max|D2f(x)|
# Para Simpson:  |E_S| <= ((b-a)^5 / (180n^4)) * max|D4f(x)|

# Derivadas simbólicas aproximadas (numéricamente con numpy)
d2_f2 = lambda x: -(np.sin(np.log(3*x)) + np.cos(np.log(3*x))) / (x**2) #derivada segunda de f3
d4_f2 = lambda x: (8*np.sin(np.log(3*x)) + 7*np.cos(np.log(3*x))) / (x**4) #derivada cuarta de f2

# Evaluamos máximo absoluto en el intervalo
xfino = np.linspace(a, b, 1000)
#busco valores maximos de la  derivada
max_f2 = np.max(np.abs(d2_f2(xfino))) 
max_f4 = np.max(np.abs(d4_f2(xfino)))

E_trapecio = ((b-a)**3 / (12 * n**2)) * max_f2
E_simpson  = ((b-a)**5 / (180 * n**4)) * max_f4

print("\nCotas de error teóricas:")
print(f"Trapecio: ≤ {E_trapecio:.2e}")
print(f"Simpson:  ≤ {E_simpson:.2e}")

# 5️⃣ Gráfico
plt.plot(xfino, f2(xfino), 'r', label="f(x)")
plt.fill_between(xfino, 0, f2(xfino), where=(xfino>=a)&(xfino<=b), color='lightblue', alpha=0.5)
plt.axhline(0, color='k', linestyle='--')
plt.title("Área bajo f(x) = sin(ln(3x)) en [1, x0]")
plt.legend()
plt.grid(True)
plt.show()'''

#==========================================================================================================
#EJERCICIO 3
t = [0,1,2,3,5,7]
V = [1,3,8,7,2,7]

# Grado máximo permitido: 5 (n = len(t) - 1)
coef = np.polyfit(t, V, 5)
P = np.poly1d(coef)

# Imprimir el polinomio para poder deriivarlo luego
print("Polinomio ajustado:")
print(P)

a = min(t)
b = max(t)

# Puntos para graficar el ajuste suave
tt = np.linspace(a, b, 100)
plt.plot(t, V, 'ro', label='Datos')
plt.plot(tt, P(tt), 'c', label='Ajuste polinomial grado 5')
plt.legend()
plt.grid(True)
plt.show()

#AHORA ME PIDEN EL AREA BAJO ESTA CURVA CON 8 NODOS 
#ES TEMA ES QUE SON n=7 INTERVALOS Y NPO PUEDO USAR SIMPSON

#trapacio solo
from funciones import fc_TrapreciosCompuesto
n = 7
Itrap = fc_TrapreciosCompuesto(P, a, b, n)
print(f"Área por trapecio compuesto (8 nodos): {Itrap:.6f}")

# 4️⃣ Cota de error teórica

# Para trapecio: |E_T| <= ((b-a)^3 / (12n^2)) * max|D2f(x)|

# Derivadas simbólicas aproximadas (numéricamente con numpy)
dP = P.deriv()   # ← derivada simbólica
print(dP)        # te muestra el polinomio derivado
valor = dP(2)    
print(dP)
# Evaluamos máximo absoluto en el intervalo
xfino = np.linspace(a, b, 1000)
#busco valores maximos de la  derivada
max_dP = np.max(np.abs(dP(xfino))) 
#max_f4 = np.max(np.abs(d4_f2(xfino)))

E_trapecio = ((b-a)**3 / (12 * n**2)) * max_dP
#E_simpson  = ((b-a)**5 / (180 * n**4)) * max_f4

print("\nCotas de error teóricas:")
print(f"Trapecio: ≤ {E_trapecio:.2e}")
#print(f"Simpson:  ≤ {E_simpson:.2e}")

# 5️⃣ Gráfico
plt.plot(xfino, P(xfino), 'r', label="f(x)")
plt.fill_between(xfino, 0, P(xfino), where=(xfino>=a)&(xfino<=b), color='lightblue', alpha=0.5)
plt.axhline(0, color='k', linestyle='--')
plt.title("Área bajo f(x) = P en [1, x0]")
plt.legend()
plt.grid(True)
plt.show()
