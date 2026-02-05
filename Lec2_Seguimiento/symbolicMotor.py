import sympy as sp

# Define las variables simbólicas
s = sp.symbols('s')
km, L, R, J, B, kb = sp.symbols('km L R J B kb')

# Define las funciones de transferencia
tf1 = km / (L * s + R)
tf2 = 1 / (J * s + B)

# Define la función de transferencia de realimentación (feedback)
# La función de transferencia en lazo cerrado es G / (1 + G*H), donde G es el sistema abierto y H el feedback
# En este caso, asumimos que el feedback es simplemente kb, sin dependencia de s para simplificar
motor_tf_open = tf1 * tf2  # Sistema en lazo abierto
motor_tf = motor_tf_open / (1 + motor_tf_open * kb)

# Simplificar la función de transferencia resultante
motor_tf = sp.simplify(motor_tf)
print(motor_tf)
num, denom = motor_tf.as_numer_denom()
denom = sp.expand(denom)
motor_tf = num/denom
print(motor_tf)

ka, kt = sp.symbols('ka, kt')
motor_tf_control = sp.simplify(ka*motor_tf/(1+motor_tf*ka*kt))
print('Motor tf control: ', motor_tf_control)
# print("La función de transferencia resultante es:")
# sp.pprint(motor_tf, use_unicode=True)