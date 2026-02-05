import numpy as np
import matplotlib.pyplot as plt
import control as ctl

# Parámetros del motor
R = 2.56 
L = 0.008 * (10**-3)
km = 0.779 * (10**-3)
kb = 1/1288.05
B = 1.46 * (10**-8)
J = 0.017 * (10**-7)
K = 0.01  # Ganancia del controlador proporcional
kt = 1  # Ganancia del tacómetro

# Define transfer functions
tf1 = ctl.tf([km], [L, R])
tf2 = ctl.tf([1], [J, B])

# Sistema en lazo abierto sin controlador
sys_open_loop = tf1 * tf2
sys_open_loop = ctl.feedback(tf1*tf2, kb)

# Sistema en lazo cerrado con controlador proporcional y feedback de back emf
sys_feedback = ctl.feedback(K * sys_open_loop, kt)

# Salida del controlador
controller_output = ctl.feedback(K, kt * sys_open_loop)


# Configuración de tiempo
time = np.linspace(0, 1, 1000)  # Vector de tiempo para todas las respuestas

# Compute the step response for open loop and closed loop
t_open, response_open = ctl.step_response(sys_open_loop*0.46, T=time)
t_closed, response_closed = ctl.step_response(sys_feedback*600, T=time)
t_controller, response_controller = ctl.step_response(controller_output*600, T=time)

# Create a figure for the subplots
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

# Create a plot for the step responses

ax1.plot(t_open, response_open, label='Open-Loop Response')
ax1.plot(t_closed, response_closed, label='Closed-Loop Response')
ax1.set_title('Step Response of the Motor System')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Angular Velocity [rad/s]')
ax1.grid(True)
ax1.legend()

# Plot the controller's response
ax2.plot(t_controller, response_controller)
ax2.set_title("Controller's Response")
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Response')
ax2.grid(True)

plt.tight_layout()
plt.show()
