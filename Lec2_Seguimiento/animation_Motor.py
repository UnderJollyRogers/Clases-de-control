import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import control as ctl

# Parameters
R = 2.56 
L = 0.008 * (10**-3)
km = 0.779 * (10**-3)
kb = 1/1288.05
B = 1.46 * (10**-8)
J = 0.017 * (10**-7)
ratio = 100
# Define transfer functions
tf1 = ctl.tf([km], [L, R])
tf2 = ctl.tf([1], [J, B])

# Define the feedback transfer function
motor_tf = ctl.feedback(tf1*tf2, kb)

# Configuración de tiempo
dt = 0.1  # paso de tiempo
tf = np.array([0])

thetaInit = 0  # Ángulo inicial

# Crear la figura y el grid de subplots
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1)

# Configuración del subplot del motor (gráfico polar)
ax_motor = plt.subplot(gs[0], projection='polar')
ax_motor.set_xticklabels([])  # Quitar etiquetas del eje x
ax_motor.set_yticklabels([])  # Quitar etiquetas del eje y
ax_motor.set_ylim(0, 1)  # Limitar el radio para que el círculo se mantenga en tamaño constante
ax_motor.grid(False)  # Quitar la grilla

# Círculo que representa el motor
motor_circle = plt.Circle((0, 0), 5, color="blue", alpha=0.5)
ax_motor.add_artist(motor_circle)

# Línea que representa el eje del motor
motor_shaft, = ax_motor.plot([], [], color="black")

# Configuración del subplot de la velocidad (gráfico de línea)
ax_velocidad = plt.subplot(gs[1])
ax_velocidad.set_xlim(0, 100)
ax_velocidad.set_ylim(-20, 20)  # Ajustar según sea necesario
ax_velocidad.set_xlabel('Tiempo (s)')
ax_velocidad.set_ylabel('Velocidad angular (rad/s)')
ax_velocidad.grid(True)  # Mostrar la grilla

# Línea que representa la velocidad angular
linea_velocidad, = ax_velocidad.plot([], [], 'r-', label='Velocidad angular')

# Agregar slider para control de voltaje
ax_voltaje = plt.axes([0.2, 0.05, 0.65, 0.03])
initial_voltaje = 0
slider_voltaje = Slider(ax_voltaje, 'Voltaje', -1, 1, valinit=initial_voltaje)
y = np.array([0])
# Función de animación
def animate(i):
    global thetaInit
    global tf
    global initial_voltaje
    global y
    global dt
    voltaje_actual = slider_voltaje.val

    # Calcular la respuesta forzada del motor al voltaje actual
    tf = np.append(tf[:-1], np.linspace(tf[-1], tf[-1] + dt, 11))
    _, y_out = ctl.step_response(motor_tf*(voltaje_actual - initial_voltaje), T=tf[-11:-1])
    y = np.append(y, np.full(len(y_out),y[-1]) + y_out)
    initial_voltaje = voltaje_actual
    velocidad_angular = y[-1]/ratio  # La última velocidad angular calculada
    thetaInit += velocidad_angular * dt  # Incrementar el ángulo basado en la velocidad
    motor_shaft.set_data([0, thetaInit], [0, 0.5])  # Actualizar la posición del eje del motor
    linea_velocidad.set_data(tf, y/ratio)  # Actualizar la línea de velocidad
    return motor_shaft, linea_velocidad

# Crear la animación
ani = FuncAnimation(fig, animate, frames=1000, interval=100, blit=False)

# Mostrar la animación
plt.show()
