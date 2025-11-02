# Código 1 modificado: Guardar como archivo CSV en lugar de Excel
import numpy as np
import pandas as pd
from scipy.signal import TransferFunction, bode

# Planta simulada:
num = [8.88e-4]
den = [1, 0.466, 0.0477, 1.11e-3]
system = TransferFunction(num, den)

# Frecuencia (logspace para análisis de Bode)
frequencies = np.logspace(0, 2, 300)
w, mag, phase = bode(system, w=frequencies)

# Crear DataFrame
df = pd.DataFrame({
    'frecuencia (rad/s)': w,
    'magnitud (dB)': mag,
    'fase (grados)': phase
})

# Guardar como CSV
csv_path = 'datos_bode_generados.csv'
df.to_csv(csv_path, index=False)

csv_path  # Mostrar ruta del archivo generado para descarga o uso posterior
