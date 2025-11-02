import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, dual_annealing
from scipy.signal import TransferFunction, freqresp, bode

# Leer datos desde archivo CSV
df = pd.read_csv('datos_bode_generados.csv')
w = df['frecuencia (rad/s)'].to_numpy()
mag = df['magnitud (dB)'].to_numpy()
phase = df['fase (grados)'].to_numpy()

best_error = np.inf
best_params = None
best_order = None
best_tf = None
alpha = 10

for n_zeros in  [0]:#range(0, 3):
    for n_polos in [3]:#range(n_zeros + 1, n_zeros + 4):
        print('zeros y polos', n_zeros, n_polos)
        def construir_tf(params):
            num = params[:n_zeros + 1]
            den = np.concatenate(([1.0], params[n_zeros + 1:]))
            return num, den

        def modelo_freqresp(params, w):
            num, den = construir_tf(params)
            sys = TransferFunction(num, den)
            print(sys)
            _, mag_c, phase_c = bode(sys, w)
            return mag_c, phase_c

        def error(params):
            mag_est, phase_est = modelo_freqresp(params, w)
            e_mag = np.sum((mag_est - mag)**2)
            e_phase = np.sum((phase_est - phase)**2)
            error = e_mag +  e_phase
            print('error: ', error)
            return error

        params0 = 0.00001*np.concatenate((np.ones(n_zeros + 1), np.ones(n_polos)))
        # límites inferiores y superiores
        lower_bounds = np.concatenate((
            np.full(n_zeros + 1, -np.inf),  # ceros: sin restricciones
            np.full(n_polos, 0.000001)          # polos: positivos
        ))

        upper_bounds = np.concatenate((
            np.full(n_zeros + 1, np.inf),   # ceros: sin restricciones
            np.full(n_polos, np.inf)           # polos: máximos razonables
        ))

        bounds = Bounds(lower_bounds, upper_bounds)

        try:
            bounds = [(-10, 10)] * (n_zeros + 1) + [(1e-6, 1e3)] * n_polos
            res = dual_annealing(error, bounds)
            #res = minimize(error, params0, method='Nelder-Mead', bounds=bounds,
            #   options={'verbose': 1, 'maxiter': 1000})
            if res.success:
                if  res.fun < best_error:
                    best_error = res.fun
                    best_params = res.x
                    best_order = (n_zeros, n_polos)
                    num_best, den_best = construir_tf(res.x)
                    best_tf = TransferFunction(num_best, den_best)
        except Exception:
            continue

# Visualización y función estimada
if best_tf is not None:
    _, H_best = freqresp(best_tf, w)
    mag_fit = 20 * np.log10(np.abs(H_best))
    phase_fit = np.angle(H_best, deg=True)

    from sympy import symbols, simplify, Eq, latex
    from sympy.abc import s
    from IPython.display import display, Math

    num_fit, den_fit = best_tf.num, best_tf.den
    num_expr = sum(coef * s**i for coef, i in zip(num_fit, reversed(range(len(num_fit)))))
    den_expr = sum(coef * s**i for coef, i in zip(den_fit, reversed(range(len(den_fit)))))
    G_s = simplify(num_expr / den_expr)
    display(G_s)

    # Graficar comparación
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag, label='Datos originales')
    plt.semilogx(w, mag_fit, '--', label='Modelo ajustado')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True, which='both')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase, label='Datos originales')
    plt.semilogx(w, phase_fit, '--', label='Modelo ajustado')
    plt.xlabel('Frecuencia (rad/s)')
    plt.ylabel('Fase (grados)')
    plt.grid(True, which='both')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No se encontró un modelo factible con coeficientes positivos.")