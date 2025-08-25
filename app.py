import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  
    tf.keras.layers.Dense(units=1)      
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

print("Hagamos una predicción!")
entrada = np.array([[100.0]])  
resultado = modelo.predict(entrada)
print("El resultado es " + str(resultado[0][0]) + " fahrenheit!")
