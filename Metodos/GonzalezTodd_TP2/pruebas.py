import numpy as np
import matplotlib.pyplot as plt

# Simulamos una señal (vos usarías tus datos reales)
X = np.linspace(0, 1, 500)
Y = 0.2*np.sin(8*np.pi*X) + np.exp(-100*(X-0.3)**2) - np.exp(-100*(X-0.7)**2)

# Marcamos posiciones características (por ejemplo)
P, Q, R, S, T = 0.25, 0.29, 0.3, 0.32, 0.38

plt.plot(X, Y, 'k', linewidth=2)

# --- Líneas verticales ---
for pos, label, color in [(P,'P','purple'), (Q,'Q','green'), (R,'R','red'),
                          (S,'S','orange'), (T,'T','blue')]:
    plt.axvline(pos, color=color, linestyle='--', alpha=0.6)
    plt.text(pos, max(Y)+0.05, label, color=color, ha='center', fontsize=12, fontweight='bold')

# --- Ejemplo de marcar intervalos ---
plt.hlines(y=-0.3, xmin=Q, xmax=S, color='gold', linewidth=3, label='Complejo QRS')
plt.hlines(y=-0.35, xmin=Q, xmax=T, color='pink', linewidth=3, label='Intervalo QT')
plt.hlines(y=-0.4, xmin=P, xmax=R, color='limegreen', linewidth=3, label='Intervalo PR')

plt.legend()
plt.title("Identificación de intervalos en ECG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)
plt.show()
