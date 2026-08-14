'''
-sirve para aproximar una funcion dond solo tengo varios datos
-O para simplificar una funcion por otra que sea mas manejable --->aproximo y con el rsto de taylor se que tan preciso es
-
'''
import matplotlib.pyplot as plt
import numpy as np

x = [-2,1,4,7]
x = np.array(x)
y = [-3,2,3,1]
y = np.array(y)
plt.plot (x, y, 'o')
#plt.show()

'''
np.polyfit(para hallar el coef del polinomio interpolador)
np.polyval(para evaluar el polinomio dandole el coeficiente)
'''
'''
'co' : o en cien
'''
'''COEF1'''
coef1 = np.polyfit(x,y,1)
xfino1 = np.arange(-2, 7.1, 0.1)
yfino1 = (np).polyval(coef1, xfino1)  #EVALUA EL POLINOMIO
plt.plot(xfino1, yfino1, 'r')

'''COEF2'''
coef2 = np.polyfit(x,y,2)
xfino2 = np.arange(-2, 7.1, 0.1)
yfino2 = (np).polyval(coef2, xfino2)  #EVALUA EL POLINOMIO
plt.plot(xfino2, yfino2, 'r')

'''COEF3'''
coef3 = np.polyfit(x,y,2)
xfino3 = np.arange(-2, 7.1, 0.1)
yfino3 = (np).polyval(coef3, xfino3)  #EVALUA EL POLINOMIO
plt.plot(xfino3, yfino3, 'r')


'''COEF5'''
coef5 = np.polyfit(x,y,5)

#lo que imprime es (a b c d ) los coef de un pol grado 3 porque pose (x y 3)
xfino5 = np.arange(-2, 7.1, 0.1)
yfino5 = (np).polyval(coef5, xfino5)  #EVALUA EL POLINOMIO
plt.plot(xfino5, yfino5, 'c')

#ejes coordenados
plt.axhline(0, color='k', linestyle="-")  # eje x
plt.axvline(0, color='k', linestyle="-")  # eje y

plt.show()


'''
puedo poner:
- 1 valor como f(x)
-1 vector "mas fino" para ggraficar
-x fino = (primer x: ultimo x : un direnciaL chiquito (0,1)) 
LO QUE HACES ES UNIRTE LOS PUNTOS CON UNA LINEA CONTINUA)
(GRAFICA DE DE 0,1)
-
)
-
'''