import matplotlib.pyplot as plt
import numpy as np

x = (-10, 10.5, 0.5)
n = len(x)
y = 1/ (1 + (x*x))

coef = (np).polyfit(x, y, n-1)

xfino = [-10, 10.1 , 0.1]
yfino = (np).polyval(coef, xfino)

plt.plot(xfino, yfino, 'c')
plt.plot(x,y,'g')
plt.show


''''BUSCAR LO DE SPLINE EN EL CAMPUS 
ALLGO DEL PATO EN VUELO'''