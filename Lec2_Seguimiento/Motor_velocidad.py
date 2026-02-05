import numpy as np
import matplotlib.pyplot as plt
import control as ctl

time = np.linspace(0, 0.1, 1000)  # Time vector for all responses

# Parameters
R = 2.56 
L = 0.008 * (10**-3)
km = 0.779 * (10**-3)
kb = 1/1288.05
B = 1.46 * (10**-8)
J = 0.017 * (10**-7)

# Define transfer functions
tf1 = ctl.tf([km], [L, R])
tf2 = ctl.tf([1], [J, B])

# Define the feedback transfer function
sys_feedback = ctl.feedback(tf1*tf2, kb)

# Compute the step response
t_out, response = ctl.step_response(0.494*sys_feedback, T=time)

# Create a plot for the step response
plt.figure(figsize=(10, 6))
plt.plot(t_out, response)
plt.title('Step Response of the Feedback System')
plt.xlabel('Time [s]')
plt.ylabel('Velocidad angular [rad/s]')
plt.grid(True)
plt.show()

# Print the feedback system for reference
print("Feedback System:", sys_feedback)
