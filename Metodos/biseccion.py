import numpy as np
import matplotlib.pyplot as plt

#BISECCIÓN
 
def fcbiseccion (f, inter, tol):
#defino funcion, lambda x: significa f(x)=
   
    #defin intervalo
    a= inter [0]
    b= inter [1]

#veo si el int es valido
    if f(a)*f(b)<0:


    #busco el punto medio del int

        x0=(a+b)/2
        r=(b-a)/2

    #para que me de el error dentro de la tol que quiero
        while r>tol:
        
            if f(a)*f(x0)<0:
                b=x0
            else: a=x0
        
            
            x0=(a+b)/2
            r=(b-a)/2

        print(f"la raiz es:{x0}")
        print(f"el error es:{r}")

    else: print ("no sirve el intervalo")

    # gráfico
    x = np.linspace(-2, 10, 100)
    y1 = f(x)
    plt.plot(x, y1, 'r', label="f(x)")  # r=rojo

    # Ejes coordenados
    plt.axhline(0, color='k', linestyle='--')  # eje x
    plt.axvline(0, color='k', linestyle='--')  # eje y

    # Opcional: leyenda y grilla
    plt.legend()
    plt.grid(True)

    # agrega punto en la raiz 
    plt.plot(x0, f(x0), 'bo', label='Raíz')  # punto azul en la raíz

    plt.show()
    
