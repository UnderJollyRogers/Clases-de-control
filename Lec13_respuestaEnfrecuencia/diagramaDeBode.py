import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode

# Definir función de transferencia: 
num = [8.88e-4]
den = [1, 0.466, 0.0477, 1.11e-3]
system = TransferFunction(num, den)
print(system)
# Obtener respuesta en frecuencia
w, mag, phase = bode(system)

# Graficar diagrama de Bode
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Subgráfico de magnitud
ax1.semilogx(w, mag)
ax1.set_ylabel('Magnitud (dB)')
ax1.set_title('Diagrama de Bode')
ax1.grid(True, which='both')  # Grilla principal y secundaria
ax1.minorticks_on()

# Subgráfico de fase
ax2.semilogx(w, phase)
ax2.set_xlabel('Frecuencia (rad/s)')
ax2.set_ylabel('Fase (grados)')
ax2.grid(True, which='both')  # Grilla principal y secundaria
ax2.minorticks_on()

plt.tight_layout()
plt.show()